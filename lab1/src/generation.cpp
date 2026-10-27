#include "generation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif


// ============================================================
// Fenwick Tree / Binary Indexed Tree
// Supports:
//   - point updates: O(log N)
//   - prefix sums:   O(log N)
//   - weighted random selection: O(log N)
// ============================================================
class FenwickTree {
public:
    explicit FenwickTree(int n)
        : n_(n), tree_(n + 1, 0) {}

    void add(int index, int64_t value) {
        // Convert 0-based index to 1-based.
        ++index;

        while (index <= n_) {
            tree_[index] += value;
            index += index & -index;
        }
    }

    int64_t sum(int index) const {
        // Sum [0, index].
        if (index < 0)
            return 0;

        ++index;

        int64_t result = 0;

        while (index > 0) {
            result += tree_[index];
            index -= index & -index;
        }

        return result;
    }

    int64_t total() const {
        return sum(n_ - 1);
    }

    // Returns the smallest index whose prefix sum is > target.
    //
    // target must satisfy:
    //     0 <= target < total()
    //
    int find_by_order(int64_t target) const {
        int index = 0;

        // Largest power of 2 <= n.
        int bit = 1;
        while ((bit << 1) <= n_)
            bit <<= 1;

        for (; bit != 0; bit >>= 1) {
            int next = index + bit;

            if (next <= n_ && tree_[next] <= target) {
                index = next;
                target -= tree_[next];
            }
        }

        // index is the number of elements before the answer.
        // Convert from Fenwick's 1-based representation.
        return index;
    }

private:
    int n_;
    std::vector<int64_t> tree_;
};


std::vector<std::vector<int>> generate_followers_matrix(int num_users) {
    if (num_users <= 0) {
        return {};
    }


    std::vector<std::vector<int>> matrix(
        num_users,
        std::vector<int>(num_users, 0)
    );


    std::random_device rd;
    std::mt19937 gen(rd());


    const int initial_links =
        std::min(
            num_users,
            std::max(3, num_users / 200)
        );

    const int num_supercelebs =
        std::min(
            num_users,
            std::max(1, num_users / 500)
        );

    const int superceleb_weight =
        1000 + static_cast<int>(gen() % 5000);

    const int avg_follows_per_user =
        std::max(10, num_users / 100);

    std::exponential_distribution<double> follow_dist(
        1.0 / static_cast<double>(avg_follows_per_user)
    );

    // Degree / preferential attachment weights
    //
    // Use int64_t because super-celebrity weights plus a large
    // number of edges can exceed a normal int.
    std::vector<int64_t> degree(num_users, 1);

    // Fenwick tree stores current degree weights.
    FenwickTree weights(num_users);

    for (int i = 0; i < num_users; ++i) {
        weights.add(i, 1);
    }

    // Give super-celebs huge initial weight.

    for (int i = 0; i < num_supercelebs; ++i) {
        degree[i] += superceleb_weight;
        weights.add(i, superceleb_weight);
    }

    // Seed a small fully-connected core.
    //
    // Matrix writes are independent, so this part can be
    // parallelized safely.

    #pragma omp parallel for schedule(static)
    for (int u = 0; u < initial_links; ++u) {
        for (int v = 0; v < initial_links; ++v) {
            if (u != v) {
                matrix[u][v] = 1;
            }
        }
    }

    // Update degrees after the parallel matrix initialization.
    //
    // Each core user receives initial_links - 1 followers.
    for (int v = 0; v < initial_links; ++v) {
        degree[v] += initial_links - 1;
        weights.add(v, initial_links - 1);
    }

    // Add new users.

    for (int u = initial_links; u < num_users; ++u) {

        int follows_to_make =
            std::max(
                1,
                static_cast<int>(follow_dist(gen))
            );

        if (follows_to_make > u) {
            follows_to_make = u;
        }

        for (int edge = 0; edge < follows_to_make; ++edge) {

            // We only want users [0, u).
            //
            // The Fenwick tree contains weights for future
            // users as well, so get the total weight of the
            // valid prefix.
            const int64_t available_weight =
                weights.sum(u - 1);

            if (available_weight <= 0) {
                break;
            }

            std::uniform_int_distribution<int64_t> dist(
                0,
                available_weight - 1
            );

            const int64_t random_value = dist(gen);

            int v = weights.find_by_order(random_value);

            // Safety check.
            if (v < 0 || v >= u) {
                continue;
            }

            // Prevent duplicate edges.
            const int64_t old_degree = degree[v];

            weights.add(v, -old_degree);

            matrix[u][v] = 1;

            ++degree[v];

            weights.add(v, degree[v]);
        }
    }

    return matrix;
}


// Generate user features.
std::vector<std::vector<double>>
generate_user_features(int num_users, int feature_dim) {

    if (num_users <= 0 || feature_dim <= 0) {
        return {};
    }

    std::vector<std::vector<double>> features(
        num_users,
        std::vector<double>(feature_dim)
    );

    #pragma omp parallel
    {
        int thread_id = 0;

        #ifdef _OPENMP
        thread_id = omp_get_thread_num();
        #endif

        // Create a deterministic but different seed for each
        // worker.
        std::mt19937 gen(
            123u +
            static_cast<unsigned int>(thread_id) * 1000003u
        );

        std::uniform_real_distribution<double> dist(0.0, 1.0);

        #pragma omp for schedule(static)
        for (int u = 0; u < num_users; ++u) {
            for (int f = 0; f < feature_dim; ++f) {
                features[u][f] = dist(gen);
            }
        }
    }

    return features;
}