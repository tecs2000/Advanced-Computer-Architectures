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
                                const std::vector<std::vector<double>> &aggregated_features,
                                const std::vector<int> &follower_counts)
{

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

struct Result
{
    int user_id;
    std::vector<double> features;
};

// ----------------------
// Worker function
// ----------------------
// Worker function: compute aggregated follower features for a user
void worker_function(
    std::queue<Task> &tasks,
    const std::vector<std::vector<int>> &followers_matrix,
    const std::vector<std::vector<double>> &user_features,
    std::vector<Result> &private_results,
    const std::vector<int> &follower_counts)
{
    
    int feature_dim = user_features[0].size();
    int n = (int)followers_matrix[0].size();

    while (!tasks.empty())
    {

        Task task = tasks.front();
        tasks.pop();

        int user_id = task.user_id;

        // Compute aggregated_features[user_id] = followers_matrix[user_id] × user_features
        std::vector<double> feature_sum(feature_dim, 0.0);
        double total_follows = follower_counts[user_id];

        for (int j = 0; j < feature_dim; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < n; k++)
            {
                sum += followers_matrix[user_id][k] *
                       user_features[k][j];
            }

            if (total_follows > 0)
                feature_sum[j] = sum / total_follows;
        }

        // Store the result in this thread's private output.
        private_results.push_back({user_id, std::move(feature_sum)});
    }
}

void master_function(
    int num_users,
    int num_worker_threads,
    std::vector<std::queue<Task>> &tasks_per_thread,
    const std::vector<std::vector<int>> &followers_matrix,
    const std::vector<std::vector<double>> &user_features,
    const std::vector<int> &follower_counts,
    std::vector<std::thread> &workers,
    std::vector<std::vector<Result>> &private_results)
{

    // Exe 1.4 - Distribute tasks using contiguous chunks
    for (int t = 0; t < num_worker_threads; ++t)
    {
        int start = t * num_users / num_worker_threads;
        int end = (t + 1) * num_users / num_worker_threads;

        // Avoid reallocations while workers append results
        private_results[t].reserve(end - start);

        for (int u = start; u < end; ++u)
            tasks_per_thread[t].push({u});
    }

    // Spawn worker threads.
    for (int t = 0; t < num_worker_threads; ++t)
    {
        workers.emplace_back(
            worker_function,
            std::ref(tasks_per_thread[t]),
            std::cref(followers_matrix),
            std::cref(user_features),
            std::ref(private_results[t]),
            std::cref(follower_counts));
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
    auto follower_counts = compute_follower_counts(followers_matrix); // Compute follower counts
    
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
    std::vector<std::vector<Result>> private_results(num_worker_threads);
    
    // Shared synchronization primitives:
    std::vector<std::thread> workers;
    
    // Timer
    auto start_time = std::chrono::high_resolution_clock::now();
    
    master_function(
        num_users,
        num_worker_threads,
        tasks_per_thread,
        followers_matrix,
        user_features,
        follower_counts,
        workers,
        private_results
    );
    
    // Join all worker threads (wait for them to finish)
    for (auto &th : workers)
        th.join();
    
    // Concatenate the results
    std::vector<std::vector<double>> aggregated_features(num_users, std::vector<double>(feature_dim, 0.0));
    for (const auto &thread_results : private_results)
    {
        for (const auto &result : thread_results)
        {
            aggregated_features[result.user_id] = result.features;
        }
    }

    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Show the result
    std::cout << "\nTotal worker computation time: " << elapsed.count() << " seconds\n";
    print_top_and_bottom_users(followers_matrix, aggregated_features, follower_counts);

    return 0;
}
