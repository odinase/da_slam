#ifndef DA_SLAM_VISUALIZATION_COLORS_HPP
#define DA_SLAM_VISUALIZATION_COLORS_HPP

#include "imgui.h"

namespace da_slam::visualization::colors
{

constexpr ImVec4 rgb2float(int r, int g, int b, float a)
{
    constexpr float max_rgb_range_float = 255.0;
    return ImVec4(static_cast<float>(r) / max_rgb_range_float, float(g) / max_rgb_range_float,
                  float(b) / max_rgb_range_float, a);
}

constexpr ImVec4 CLEAR = ImVec4(0.45F, 0.55F, 0.6F, 1.0F);
constexpr ImVec4 RED = ImVec4(1.0F, 0.0F, 0.0F, 1.0F);
constexpr ImVec4 GREEN = ImVec4(0.0F, 1.0F, 0.0F, 1.0F);
constexpr ImVec4 BLUE = ImVec4(0.0F, 0.0F, 0.1F, 1.0F);
constexpr ImVec4 UGLY_YELLOW = ImVec4(185.0F / 255.0F, 185.0F / 255.0F, 72.0F / 255.0F, 1.0F);

constexpr ImVec4 YELLOW = ImVec4(1.0F, 1.0F, 0.0F, 1.0F);
constexpr ImVec4 CYAN = ImVec4(0.0F, 1.0F, 1.0F, 1.0F);
constexpr ImVec4 MAGENTA = ImVec4(1.0F, 0.0F, 1.0F, 1.0F);

constexpr ImVec4 NAVY = ImVec4(0.0F, 0.0F, 0.5F, 1.0F);

constexpr ImVec4 ORANGE = ImVec4(1.0F, 140.0F / 255.0F, 0.0F, 1.0F);
constexpr ImVec4 ROSY_BROWN = ImVec4(188.0F / 255.0F, 143.0F / 255.0F, 143.0F / 255.0F, 1.0F);

}  // namespace da_slam::visualization::colors

#endif  // DA_SLAM_VISUALIZATION_COLORS_HPP