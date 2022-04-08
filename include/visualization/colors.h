#pragma once

#include "imgui.h"

namespace visualization
{
    namespace colors
    {
        constexpr ImVec4 CLEAR = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
        constexpr ImVec4 RED = ImVec4(1.0f, 0.0f, 0.0f, 1.00f);
        constexpr ImVec4 GREEN = ImVec4(0.0, 1.0, 0.0f, 1.00f);
        constexpr ImVec4 BLUE = ImVec4(0.0f, 0.0f, 0.1f, 1.00f);
        constexpr ImVec4 UGLY_YELLOW = ImVec4(185.0 / 255.0, 185.0 / 255.0, 72.0 / 255.0, 1.00f);
    } // namespace colors
} // namespace visualization