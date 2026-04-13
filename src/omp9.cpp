#include "common.hpp"

using Matrix = std::vector<int>;

int max_row_mins_flat(const Matrix& m, std::size_t n) {
    int result = std::numeric_limits<int>::min();
    #pragma omp parallel for reduction(max:result)
    for (std::size_t i = 0; i < n; ++i) {
        int row_min = std::numeric_limits<int>::max();
        const std::size_t off = i * n;
        for (std::size_t j = 0; j < n; ++j) row_min = std::min(row_min, m[off + j]);
        result = std::max(result, row_min);
    }
    return result;
}

int max_row_mins_nested(const Matrix& m, std::size_t n, int total_threads) {
    const int outer_threads = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(total_threads))));
    const int inner_threads = std::max(1, total_threads / outer_threads);
    int result = std::numeric_limits<int>::min();
    #pragma omp parallel num_threads(outer_threads) reduction(max:result)
    {
        const int tid = omp_get_thread_num();
        const int team_size = omp_get_num_threads();
        const std::size_t row_begin = (n * static_cast<std::size_t>(tid)) / static_cast<std::size_t>(team_size);
        const std::size_t row_end = (n * static_cast<std::size_t>(tid + 1)) / static_cast<std::size_t>(team_size);

        int outer_local_max = std::numeric_limits<int>::min();
        std::vector<int> partial_mins(static_cast<std::size_t>(inner_threads), std::numeric_limits<int>::max());

        #pragma omp parallel num_threads(inner_threads) shared(partial_mins, outer_local_max)
        {
            const int inner_tid = omp_get_thread_num();
            for (std::size_t i = row_begin; i < row_end; ++i) {
                int local_min = std::numeric_limits<int>::max();
                const std::size_t off = i * n;
                for (std::size_t j = static_cast<std::size_t>(inner_tid); j < n; j += static_cast<std::size_t>(inner_threads)) {
                    local_min = std::min(local_min, m[off + j]);
                }
                partial_mins[static_cast<std::size_t>(inner_tid)] = local_min;

                #pragma omp barrier
                #pragma omp single
                {
                    int row_min = std::numeric_limits<int>::max();
                    for (int p = 0; p < inner_threads; ++p) row_min = std::min(row_min, partial_mins[static_cast<std::size_t>(p)]);
                    outer_local_max = std::max(outer_local_max, row_min);
                }
                #pragma omp barrier
            }
        }

        result = std::max(result, outer_local_max);
    }
    return result;
}

int main(int argc, char** argv) {
    const auto args = parse_cli(argc, argv);
    if (args.size == 0 || args.mode.empty()) return 1;

    const std::size_t n = args.size;
    Matrix m(n * n);
    std::mt19937_64 rng(args.seed);
    std::uniform_int_distribution<int> dist(0, 1000000);
    for (auto& v : m) v = dist(rng);

    int result = 0;
    double total = 0.0;

    if (args.mode == "nested") omp_set_max_active_levels(2);
    else omp_set_max_active_levels(1);
    omp_set_dynamic(0);

    for (int r = 0; r < args.runs; ++r) {
        total += measure_seconds([&] {
            if (args.mode == "flat") result = max_row_mins_flat(m, n);
            else if (args.mode == "nested") result = max_row_mins_nested(m, n, args.threads);
            else std::exit(1);
        });
    }

    output_csv_row({argv[0], args.mode, args.threads, total / args.runs, std::to_string(result)});
    return 0;
}
