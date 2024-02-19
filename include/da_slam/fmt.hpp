#ifndef DA_SLAM_FMT_HPP
#define DA_SLAM_FMT_HPP

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <gtsam/inference/Symbol.h>

#include <Eigen/Core>
#include <iostream>

namespace fmt
{

// template <typename Derived>
// struct formatter<Eigen::EigenBase<Derived>> : formatter<std::string_view>
// {
//     auto format(Eigen::EigenBase<Derived>&& m, format_context& ctx) const
//     {
//         const std::stringstream ss{};
//         ss << m;
//         const std::string s = ss.str();
//         return formatter<std::string_view>::format(s, ctx);
//     }
// };

// template <>
// struct fmt::formatter<gtsam::Symbol>
// {
//     // Parses format specifications of the form ['f' | 'e'], which are not used in this example
//     constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin())
//     {
//         // No format specifiers to parse, so return the end of the range.
//         auto it = ctx.begin(), end = ctx.end();
//         // Check if there are any characters in the format string; we don't expect any.
//         if (it != end && *it != '}') {
//             throw fmt::format_error("invalid format");
//         }
//         return it;
//     }

//     template <typename FormatContext>
//     auto format(const gtsam::Symbol& symbol, FormatContext& ctx) -> decltype(ctx.out())
//     {
//         std::ostringstream oss;
//         oss << symbol;  // Use the ostream implementation of gtsam::Symbol
//         return format_to(ctx.out(), "{}", oss.str());
//     }
// };

template <typename Derived>
struct formatter<Eigen::MatrixBase<Derived>> : ostream_formatter
{
};

template <>
struct formatter<gtsam::Symbol> : ostream_formatter
{
};

}  // namespace fmt

#endif  // DA_SLAM_FMT_HPP

// #ifndef HYBRID_FG_UTILS_FMT_HPP
// #define HYBRID_FG_UTILS_FMT_HPP

// #include <fmt/format.h>
// #include <fmt/ostream.h>
// #include <fmt/ranges.h>
// #include <gtsam/inference/Symbol.h>
// #include <spdlog/spdlog.h>

// #include <Eigen/Core>
// #include <iostream>

// namespace fmt
// {
// #ifdef __linux__

// template <typename Derived>
// struct formatter<Eigen::EigenBase<Derived>> : formatter<std::string_view>
// {
//     auto format(Eigen::EigenBase<Derived>&& m, format_context& ctx) const
//     {
//         const std::stringstream ss{};
//         ss << m;
//         const std::string s = ss.str();
//         return formatter<std::string_view>::format(s, ctx);
//     }
// };

// template <>
// struct fmt::formatter<gtsam::Symbol>
// {
//     // Parses format specifications of the form ['f' | 'e'], which are not used in this example
//     constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin())
//     {
//         // No format specifiers to parse, so return the end of the range.
//         auto it = ctx.begin(), end = ctx.end();
//         // Check if there are any characters in the format string; we don't expect any.
//         if (it != end && *it != '}') {
//             throw fmt::format_error("invalid format");
//         }
//         return it;
//     }

//     template <typename FormatContext>
//     auto format(const gtsam::Symbol& symbol, FormatContext& ctx) -> decltype(ctx.out())
//     {
//         std::ostringstream oss;
//         oss << symbol;  // Use the ostream implementation of gtsam::Symbol
//         return format_to(ctx.out(), "{}", oss.str());
//     }
// };

// // template <>
// // struct formatter<gtsam::Values>
// // {
// //     auto format(gtsam::Values&& m, format_context& ctx) -> decltype(ctx.out()) const
// //     {
// //         std::string s{};
// //         for (auto&& [k, v] : m) {
// //             s += fmt::format("{}: {}", k, v);
// //         }
// //         return format_to(ctx.out(), "{}", s);
// //     }
// // };

// template <typename... T>
// FMT_INLINE void println(std::FILE* f, format_string<T...> fmt, T&&... args)
// {
//     return fmt::print(f, "{}\n", fmt::format(fmt, std::forward<T>(args)...));
// }

// template <typename... T>
// FMT_INLINE void println(format_string<T...> fmt, T&&... args)
// {
//     return fmt::println(stdout, fmt, std::forward<T>(args)...);
// }

// #elif __APPLE__

// template <typename Derived>
// struct formatter<Eigen::MatrixBase<Derived>> : ostream_formatter
// {
// };

// template <>
// struct formatter<gtsam::Symbol> : ostream_formatter
// {
// };

// #endif

// }  // namespace fmt

// #endif  // HYBRID_FG_UTILS_FMT_HPP
