#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <random>
#include <algorithm>
#include <chrono>
#include <cassert>

#include "generation.h"

// Compute the follower counts for each user
std::vector<int> compute_follower_counts(const std::vector<std::vector<int>> &followers_matrix)
{
    std::vector<int> follower_counts(followers_matrix.size(), 0);
    for (size_t u = 0; u < followers_matrix.size(); u++)
    {
        // Count how many 1's (followers) this user has
        int count = 0;
        for (int val : followers_matrix[u])
        {
            if (val == 1)
                count++;
        }
        follower_counts[u] = count;
    }
    return follower_counts;
}

// Print top 5 and bottom 5 users based on follower counts
void print_top_and_bottom_users(const std::vector<std::vector<int>> &followers_matrix,
                                const std::vector<std::vector<double>> &aggregated_features)
{
    // Compute follower counts
    auto follower_counts = compute_follower_counts(followers_matrix);

    // Create a vector of (user_id, count) pairs
    std::vector<std::pair<int, int>> user_counts;
    for (size_t u = 0; u < follower_counts.size(); u++)
    {
        user_counts.emplace_back(u, follower_counts[u]);
    }

    // Sort descending by follower count
    std::sort(user_counts.begin(), user_counts.end(),
              [](const auto &a, const auto &b)
              { return a.second > b.second; });

    int n = user_counts.size();

    std::cout << "\n=== Top 5 Users with Most Followers ===\n";
    for (int i = 0; i < std::min(5, n); i++)
    {
        int u = user_counts[i].first;
        std::cout << "User " << u << " (followers: " << user_counts[i].second << "): ";
        for (double val : aggregated_features[u])
        {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n=== Bottom 5 Users with Least Followers ===\n";
    for (int i = 0; i < std::min(5, n); i++)
    {
        int u = user_counts[n - 1 - i].first;
        std::cout << "User " << u << " (followers: " << user_counts[n - 1 - i].second << "): ";
        for (double val : aggregated_features[u])
        {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}

// ----------------------
// Task structure
// ----------------------
// Each task corresponds to processing one user row (their followers)
struct Task
{
    int user_id; // which user to process
};

// ----------------------
// Worker function
// ----------------------
// Worker function: compute aggregated follower features for a user
void worker_function(
    std::queue<Task> &tasks,
    const std::vector<std::vector<int>> &followers_matrix, // Matrix A
    const std::vector<std::vector<double>> &user_features, // Matrix B
    std::vector<std::vector<double>> &aggregated_features  // Matrix C can be upgrade for thread private
)
{
    int feature_dim = (int)user_features[0].size(); // ncolumns Matrix B

    assert((int)followers_matrix[0].size() == (int)user_features.size());
    int n = (int)followers_matrix[0].size();

    while (!tasks.empty())
    {

        Task task = tasks.front();
        tasks.pop();

        int user_id = task.user_id;

        // WCompute aggregated_features[user_id] = followers_matrix[user_id] × user_features
        std::vector<double> feature_sum(feature_dim, 0.0);
        double total_follows = compute_follower_counts(followers_matrix)[user_id]; // count of follows (used to normalize later)

        // TODO: Fix numerical precision
        for (int j = 0; j < feature_dim; j++) // all columns of Matrix B
        {
            double sum = 0;
            for (int k = 0; k < n; k++)       // all indexes
            {
                sum += followers_matrix[user_id][k] * user_features[k][j];
            }

            if (total_follows > 0.0)
            {
                feature_sum[j] = sum / total_follows;
            }
        }

        // Compute average feature vector of followed users (if any)
        aggregated_features[user_id] = feature_sum;
    }
}

void master_function(
    int num_users,
    int num_worker_threads,
    std::vector<std::queue<Task>> &tasks_per_thread,
    const std::vector<std::vector<int>> &followers_matrix,
    const std::vector<std::vector<double>> &user_features,
    std::vector<std::vector<double>> &aggregated_features,
    std::vector<std::thread> &workers)
{

    // Fill the task queues (equally distributed load balance)
    for (int u = 0; u < num_users; ++u)
    {
        int t = u % num_worker_threads;
        tasks_per_thread[t].push({u});

        std::cout << "Master assigned user " << u
                      << " to worker " << t << '\n';
    }

    // Spawn worker threads
    for (int t = 0; t < num_worker_threads; ++t)
    {
        workers.emplace_back(
            worker_function,
            std::ref(tasks_per_thread[t]),
            std::cref(followers_matrix),
            std::cref(user_features),
            std::ref(aggregated_features));
    }
}

// ----------------------
// Main
// ----------------------
int main(int argc, char *argv[])
{
    // Default values
    int num_users = 20;         // default small network
    int feature_dim = 3;        // features: [activity, likes, posts]
    int num_worker_threads = 4; // default number of worker threads

    // If the user gave arguments, override defaults
    if (argc >= 2)
    {
        num_users = std::stoi(argv[1]); // first argument = number of users
    }
    if (argc >= 3)
    {
        num_worker_threads = std::stoi(argv[2]); // second argument = number of worker threads
    }

    std::cout << "Running with " << num_users << " users and "
              << num_worker_threads << " worker threads.\n";

    // Generate random followers matrix and user features
    auto followers_matrix = generate_followers_matrix(num_users);
    auto user_features = generate_user_features(num_users, feature_dim);
    std::vector<std::vector<double>> aggregated_features(num_users, std::vector<double>(feature_dim, 0.0));

    std::cout << "Finished matrice generation" << "\n";

    std::cout << "Followers Matrix:" << "\n";
    for (int i = 0; i < num_users; i++)
    { // nlines
        for (int j = 0; j < user_features[0].size(); j++)
        { // ncols
            std::cout << followers_matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Feature Dim:" << "\n";
    for (int i = 0; i < 3; i++)
    { // nlines
        for (int j = 0; j < user_features[0].size(); j++)
        { // ncols
            std::cout << user_features[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Tasks queue
    std::vector<std::queue<Task>> tasks_per_thread(num_worker_threads);

    // Shared synchronization primitives:
    std::vector<std::thread> workers;

    // Timer
    auto start_time = std::chrono::high_resolution_clock::now();

    master_function(num_users, num_worker_threads,
                    tasks_per_thread,
                    followers_matrix, user_features,
                    aggregated_features, workers);

    // Join all worker threads (wait for them to finish)
    for (auto &th : workers)
        th.join();

    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Show the result
    std::cout << "\nTotal worker computation time: " << elapsed.count() << " seconds\n";
    print_top_and_bottom_users(followers_matrix, aggregated_features);

    return 0;
}
