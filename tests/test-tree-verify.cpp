// token tree verification (llama_batch_ext_set_tree_parent, llama_memory_tree_accept) against plain chain decodes
// usage: test-tree-verify -m model.gguf [-fa on -ctk q8_0 -ctv q8_0], needs LLAMA_RS_RECOMPUTE=1 for recurrent models

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const char * what) {
    printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
    n_fail += ok ? 0 : 1;
}

struct node {
    llama_token tok;
    int32_t     parent;
};

static std::vector<int32_t> path_to(const std::vector<node> & tree, int32_t leaf) {
    std::vector<int32_t> p;
    for (int32_t i = leaf; i >= 0; i = tree[i].parent) {
        p.push_back(i);
    }
    std::reverse(p.begin(), p.end());
    return p;
}

static bool decode_tree(llama_context * ctx, const std::vector<node> & tree, llama_pos pos0, bool as_tree) {
    common_batch batch(ctx);
    std::vector<int32_t> depth(tree.size(), 0);
    for (size_t i = 0; i < tree.size(); ++i) {
        depth[i] = tree[i].parent < 0 ? 0 : depth[tree[i].parent] + 1;
        batch.add(tree[i].tok, pos0 + depth[i], 0, true);
        if (as_tree) {
            batch.set_tree_parent((int32_t) i, tree[i].parent);
        }
    }
    const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    if (rc != 0) {
        printf("  llama_process returned %d\n", rc);
    }
    return rc == 0;
}

static bool decode_chain(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos0) {
    common_batch batch(ctx);
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.add(toks[i], pos0 + (llama_pos) i, 0, true);
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

struct cmp_result {
    float  max_diff = 0.0f;
    bool   same_top = true;
    double kld      = 0.0; // KL(softmax(a) || softmax(b))
};

static cmp_result compare(const float * a, const float * b, int n) {
    cmp_result r;
    int ta = 0, tb = 0;
    for (int i = 0; i < n; ++i) {
        r.max_diff = std::max(r.max_diff, std::fabs(a[i] - b[i]));
        ta = a[i] > a[ta] ? i : ta;
        tb = b[i] > b[tb] ? i : tb;
    }
    r.same_top = ta == tb;
    double sa = 0.0, sb = 0.0;
    for (int i = 0; i < n; ++i) {
        sa += std::exp((double) a[i] - a[ta]);
        sb += std::exp((double) b[i] - b[tb]);
    }
    const double la = a[ta] + std::log(sa), lb = b[tb] + std::log(sb);
    for (int i = 0; i < n; ++i) {
        const double lpa = a[i] - la, lpb = b[i] - lb;
        r.kld += std::exp(lpa) * (lpa - lpb);
    }
    return r;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx = 4096;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();

    auto mparams = common_model_params_to_llama(params);
    llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
    if (!model) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 31;
    cparams.n_batch   = 512;
    cparams.n_ubatch  = 512;

    llama_context_ptr ctx_a(llama_init_from_model(model.get(), cparams)); // tree
    llama_context_ptr ctx_b(llama_init_from_model(model.get(), cparams)); // chain reference
    if (!ctx_a || !ctx_b) {
        return 1;
    }

    const std::string text =
        "#include <vector>\n// merge two sorted vectors into one sorted vector\n"
        "std::vector<int> merge(const std::vector<int> & a, const std::vector<int> & b) {\n"
        "    std::vector<int> out;\n    out.reserve(a.size() + b.size());\n";
    std::vector<llama_token> prompt = common_tokenize(vocab, text, true, false);
    const llama_pos n_prompt = (llama_pos) prompt.size();

    std::mt19937 rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? 42 : params.sampling.seed);

    for (int round = 0; round < 5; ++round) {
        // rounds 0, 3, 4: a chain sent as a tree, the same batch width as the chain decode, so results must match exactly
        const bool is_chain = round == 0 || round >= 3;
        llama_memory_clear(llama_get_memory(ctx_a.get()), true);
        llama_memory_clear(llama_get_memory(ctx_b.get()), true);
        if (!decode_chain(ctx_a.get(), prompt, 0) || !decode_chain(ctx_b.get(), prompt, 0)) {
            return 1;
        }

        // round 0: a chain sent as a tree (must match exactly), then random trees
        const int n_nodes = is_chain ? 8 : (round == 1 ? 16 : 31);
        std::vector<node> tree;
        tree.push_back({ prompt.back() == 0 ? 1 : (llama_token) (rng() % n_vocab), -1 });
        for (int i = 1; i < n_nodes; ++i) {
            const int32_t parent = is_chain ? i - 1 : (int32_t) (rng() % std::min(i, 6)) + std::max(0, i - 6);
            tree.push_back({ (llama_token) (rng() % n_vocab), parent });
        }
        {
            // breadth-first order: positions must not decrease within a batch
            std::vector<int32_t> depth(tree.size(), 0), order(tree.size()), remap(tree.size());
            for (size_t i = 1; i < tree.size(); ++i) {
                depth[i] = depth[tree[i].parent] + 1;
            }
            for (size_t i = 0; i < tree.size(); ++i) {
                order[i] = (int32_t) i;
            }
            std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return depth[a] < depth[b]; });
            for (size_t i = 0; i < order.size(); ++i) {
                remap[order[i]] = (int32_t) i;
            }
            std::vector<node> sorted;
            for (int32_t i : order) {
                sorted.push_back({ tree[i].tok, tree[i].parent < 0 ? -1 : remap[tree[i].parent] });
            }
            tree = sorted;
        }
        printf("round %d: %d nodes\n", round, n_nodes);

        if (!decode_tree(ctx_a.get(), tree, n_prompt, true)) {
            check(false, "tree decode");
            continue;
        }

        // leaves
        std::vector<bool> has_child(tree.size(), false);
        for (size_t i = 1; i < tree.size(); ++i) {
            has_child[tree[i].parent] = true;
        }
        std::vector<int32_t> leaves;
        for (int32_t i = 0; i < (int32_t) tree.size(); ++i) {
            if (!has_child[i]) {
                leaves.push_back(i);
            }
        }

        // pad = 0: the path alone; pad = 1: the path plus filler tokens up to the tree size, so the batch has
        // the same width as the tree batch and the same kernels run (the filler comes after the path, so it is not seen)
        std::vector<std::vector<float>> chain0; // pad = 0 logits, for the noise floor
        for (int pad = 0; pad < 2; ++pad) {
            float  worst = 0.0f;
            bool   top_ok = true;
            double kld = 0.0, kld_floor = 0.0;
            int    n_kld = 0;
            for (int32_t leaf : leaves) {
                const auto p = path_to(tree, leaf);
                std::vector<llama_token> toks;
                for (int32_t i : p) {
                    toks.push_back(tree[i].tok);
                }
                while (pad && toks.size() < tree.size()) {
                    toks.push_back((llama_token) (rng() % n_vocab));
                }
                if (!decode_chain(ctx_b.get(), toks, n_prompt)) {
                    check(false, "chain decode");
                    break;
                }
                for (size_t d = 0; d < p.size(); ++d) {
                    const float * lb = llama_get_logits_ith(ctx_b.get(), (int32_t) d);
                    const auto r = compare(llama_get_logits_ith(ctx_a.get(), p[d]), lb, n_vocab);
                    worst  = std::max(worst, r.max_diff);
                    top_ok = top_ok && r.same_top;
                    if (pad == 0) {
                        chain0.emplace_back(lb, lb + n_vocab);
                    } else {
                        kld_floor += compare(chain0[n_kld].data(), lb, n_vocab).kld;
                    }
                    kld += r.kld;
                    n_kld++;
                }
                if (!llama_memory_seq_rm(llama_get_memory(ctx_b.get()), 0, n_prompt, -1)) {
                    check(false, "chain rollback");
                    break;
                }
            }
            printf("  %zu leaves, tree vs chain%s: max logit diff %.6f, mean KLD %.7f, same top token: %s\n",
                    leaves.size(), pad ? " (same batch width)" : "", worst, kld / n_kld, top_ok ? "yes" : "no");
            if (pad) {
                printf("  noise floor, chain vs the same chain in a wider batch: mean KLD %.7f\n", kld_floor / n_kld);
            }
            if (is_chain) {
                check(worst == 0.0f, "chain as a tree is bit-identical");
            } else {
                check(kld / n_kld < 0.01, "tree logits close to chain logits");
            }
        }

        // accept the path to the deepest leaf, then continue both contexts with the same tokens
        int32_t best = leaves[0];
        for (int32_t leaf : leaves) {
            best = path_to(tree, leaf).size() > path_to(tree, best).size() ? leaf : best;
        }
        const auto p = path_to(tree, best);
        // also try paths that stop before the leaf (rounds 3, 4: shorter than the conv window)
        const size_t n_keep = round == 2 ? std::max<size_t>(1, p.size() - 2) : round == 3 ? 1 : round == 4 ? 2 : p.size();
        std::vector<int32_t> rows(p.begin(), p.begin() + n_keep);
        check(llama_memory_tree_accept(llama_get_memory(ctx_a.get()), 0, rows.data(), (int32_t) rows.size()), "tree accept");

        // reference: the accepted path as a chain; for a chain tree the whole chain, rolled back to the accepted rows
        std::vector<llama_token> toks;
        for (int32_t i : (is_chain ? p : rows)) {
            toks.push_back(tree[i].tok);
        }
        if (!decode_chain(ctx_b.get(), toks, n_prompt) ||
            (is_chain && !llama_memory_seq_rm(llama_get_memory(ctx_b.get()), 0, n_prompt + (llama_pos) rows.size(), -1))) {
            check(false, "chain decode of the accepted path");
            continue;
        }

        check(llama_memory_seq_pos_max(llama_get_memory(ctx_a.get()), 0) == llama_memory_seq_pos_max(llama_get_memory(ctx_b.get()), 0), "same pos_max after accept");

        std::vector<llama_token> next;
        for (int i = 0; i < 4; ++i) {
            next.push_back((llama_token) (rng() % n_vocab));
        }
        const llama_pos pos_next = n_prompt + (llama_pos) rows.size();
        if (!decode_chain(ctx_a.get(), next, pos_next) || !decode_chain(ctx_b.get(), next, pos_next)) {
            check(false, "decode after accept");
            continue;
        }
        float  worst_next = 0.0f;
        bool   top_next   = true;
        double kld_next   = 0.0;
        for (int i = 0; i < (int) next.size(); ++i) {
            const auto r = compare(llama_get_logits_ith(ctx_a.get(), i), llama_get_logits_ith(ctx_b.get(), i), n_vocab);
            worst_next = std::max(worst_next, r.max_diff);
            top_next   = top_next && r.same_top;
            kld_next  += r.kld / next.size();
        }
        printf("  after accepting %zu rows: max logit diff %.6f, mean KLD %.7f, same top token: %s\n", rows.size(), worst_next, kld_next, top_next ? "yes" : "no");
        if (is_chain) {
            check(worst_next == 0.0f, "state after accept is bit-identical to the chain");
        } else {
            // the recorded inputs come from a wider batch than the chain decode, so rounding may differ
            check(kld_next < 0.01, "state after accept close to the chain");
        }
    }

    printf("%s\n", n_fail == 0 ? "ALL OK" : "FAILURES");
    return n_fail == 0 ? 0 : 1;
}
