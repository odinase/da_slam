#pragma once

#include "imgui.h"

namespace visualization
{
    namespace colors
    {
        constexpr ImVec4 rgb2float(int r, int g, int b, float a) {
            return ImVec4(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, a);
        }
        constexpr ImVec4 CLEAR = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
        constexpr ImVec4 RED = ImVec4(1.0f, 0.0f, 0.0f, 1.00f);
        constexpr ImVec4 GREEN = ImVec4(0.0, 1.0, 0.0f, 1.00f);
        constexpr ImVec4 BLUE = ImVec4(0.0f, 0.0f, 0.1f, 1.00f);
        constexpr ImVec4 UGLY_YELLOW = ImVec4(185.0 / 255.0, 185.0 / 255.0, 72.0 / 255.0, 1.00f);

        constexpr ImVec4 YELLOW = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
        constexpr ImVec4 CYAN = ImVec4(0.0f, 1.0f, 1.0f, 1.0f);
        constexpr ImVec4 MAGENTA = ImVec4(1.0f, 0.0f, 1.0f, 1.0f);

        constexpr ImVec4 NAVY = ImVec4(0.0f, 0.0f, 0.5f, 1.0f);

        constexpr ImVec4 ORANGE = ImVec4(1.0f, 140.0 / 255.0, 0.0f, 1.0f);
        constexpr ImVec4 ROSY_BROWN = ImVec4(188.0 / 255.0, 143.0 / 255.0 , 143.0 / 255.0, 1.0f);
    } // namespace colors
} // namespace visualization