#ifndef GENERATION_H
#define GENERATION_H

#include <vector>

// Generate a dense followers matrix (0/1)
std::vector<std::vector<int>> generate_followers_matrix(int num_users);

// Generate random user features [activity, likes, posts, etc.]
std::vector<std::vector<double>> generate_user_features(int num_users, int feature_dim);

#endif // GENERATION_H
