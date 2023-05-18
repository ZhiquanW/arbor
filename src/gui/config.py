from typing import Union
import dearpygui.dearpygui as dpg

viewport_size = (1200, 1000)
drag_width = 200
arbor_editor_width = 220
render_window_size = (viewport_size[0] - arbor_editor_width, viewport_size[1])


def create_default_theme_imgui() -> Union[str, int]:
    with dpg.font_registry():
        # first argument ids the path to the .ttf or .otf file
        default_font = dpg.add_font(
            "/Users/zhiquan/Projects/arbor/src/gui/Roboto/Roboto-Medium.ttf", 28
        )
    dpg.set_global_font_scale(0.5)

    dpg.bind_font(default_font)

    with dpg.theme() as theme_id:
        with dpg.theme_component(0):
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 3)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, y=1)

            dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0, 255))
            dpg.add_theme_color(
                dpg.mvThemeCol_TextDisabled,
                (int(0.60 * 255), int(0.60 * 255), int(0.60 * 255), int(1.00 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_WindowBg,
                (int(0.94 * 255), int(0.94 * 255), int(0.94 * 255), int(1.00 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ChildBg,
                (int(0.00 * 255), int(0.00 * 255), int(0.00 * 255), int(0.00 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PopupBg,
                (int(1.00 * 255), int(1.00 * 255), int(1.00 * 255), int(0.98 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Border,
                (int(0.00 * 255), int(0.00 * 255), int(0.00 * 255), int(0.30 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_BorderShadow,
                (int(0.00 * 255), int(0.00 * 255), int(0.00 * 255), int(0.00 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBg,
                (int(1.00 * 255), int(1.00 * 255), int(1.00 * 255), int(1.00 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.40 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.67 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBg,
                (
                    int(0.96 * 255),
                    int(0.96 * 255),
                    int(0.96 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBgActive,
                (
                    int(0.82 * 255),
                    int(0.82 * 255),
                    int(0.82 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBgCollapsed,
                (int(1.00 * 255), int(1.00 * 255), int(1.00 * 255), int(0.51 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_MenuBarBg,
                (
                    int(0.86 * 255),
                    int(0.86 * 255),
                    int(0.86 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarBg,
                (int(0.98 * 255), int(0.98 * 255), int(0.98 * 255), int(0.53 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrab,
                (int(0.69 * 255), int(0.69 * 255), int(0.69 * 255), int(0.80 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrabHovered,
                (int(0.49 * 255), int(0.49 * 255), int(0.49 * 255), int(0.80 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrabActive,
                (
                    int(0.49 * 255),
                    int(0.49 * 255),
                    int(0.49 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_CheckMark,
                (
                    int(0.26 * 255),
                    int(0.59 * 255),
                    int(0.98 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrab,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.78 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrabActive,
                (int(0.46 * 255), int(0.54 * 255), int(0.80 * 255), int(0.60 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Button,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.40 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonHovered,
                (
                    int(0.26 * 255),
                    int(0.59 * 255),
                    int(0.98 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonActive,
                (
                    int(0.06 * 255),
                    int(0.53 * 255),
                    int(0.98 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Header,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.31 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderHovered,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.80 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderActive,
                (
                    int(0.26 * 255),
                    int(0.59 * 255),
                    int(0.98 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Separator,
                (int(0.39 * 255), int(0.39 * 255), int(0.39 * 255), int(0.62 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SeparatorHovered,
                (int(0.14 * 255), int(0.44 * 255), int(0.80 * 255), int(0.78 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SeparatorActive,
                (
                    int(0.14 * 255),
                    int(0.44 * 255),
                    int(0.80 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ResizeGrip,
                (int(0.35 * 255), int(0.35 * 255), int(0.35 * 255), int(0.17 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ResizeGripHovered,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.67 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ResizeGripActive,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.95 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Tab,
                (int(0.76 * 255), int(0.80 * 255), int(0.84 * 255), int(0.93 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabHovered,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.80 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabActive,
                (
                    int(0.60 * 255),
                    int(0.73 * 255),
                    int(0.88 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabUnfocused,
                (int(0.92 * 255), int(0.93 * 255), int(0.94 * 255), int(0.99 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabUnfocusedActive,
                (
                    int(0.74 * 255),
                    int(0.82 * 255),
                    int(0.91 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_DockingPreview,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.22 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_DockingEmptyBg,
                (
                    int(0.20 * 255),
                    int(0.20 * 255),
                    int(0.20 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotLines,
                (
                    int(0.39 * 255),
                    int(0.39 * 255),
                    int(0.39 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotLinesHovered,
                (
                    int(1.00 * 255),
                    int(0.43 * 255),
                    int(0.35 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotHistogram,
                (
                    int(0.90 * 255),
                    int(0.70 * 255),
                    int(int(0.00 * 255)),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotHistogramHovered,
                (
                    int(1.00 * 255),
                    int(0.45 * 255),
                    int(int(0.00 * 255)),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableHeaderBg,
                (
                    int(0.78 * 255),
                    int(0.87 * 255),
                    int(0.98 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableBorderStrong,
                (
                    int(0.57 * 255),
                    int(0.57 * 255),
                    int(0.64 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableBorderLight,
                (
                    int(0.68 * 255),
                    int(0.68 * 255),
                    int(0.74 * 255),
                    int(1.00 * 255),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableRowBg,
                (
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                ),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableRowBgAlt,
                (int(0.30 * 255), int(0.30 * 255), int(0.30 * 255), int(0.09 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TextSelectedBg,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.35 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_DragDropTarget,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.95 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_NavHighlight,
                (int(0.26 * 255), int(0.59 * 255), int(0.98 * 255), int(0.80 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_NavWindowingHighlight,
                (int(0.70 * 255), int(0.70 * 255), int(0.70 * 255), int(0.70 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_NavWindowingDimBg,
                (int(0.20 * 255), int(0.20 * 255), int(0.20 * 255), int(0.20 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ModalWindowDimBg,
                (int(0.20 * 255), int(0.20 * 255), int(0.20 * 255), int(0.35 * 255)),
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_FrameBg,
                (
                    int(1.00 * 255),
                    int(1.00 * 255),
                    int(1.00 * 255),
                    int(1.00 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_PlotBg,
                (
                    int(0.42 * 255),
                    int(0.57 * 255),
                    int(1.00 * 255),
                    int(0.13 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_PlotBorder,
                (
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendBg,
                (
                    int(1.00 * 255),
                    int(1.00 * 255),
                    int(1.00 * 255),
                    int(0.98 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendBorder,
                (int(0.82 * 255), int(0.82 * 255), int(0.82 * 255), int(0.80 * 255)),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendText,
                (
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(1.00 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_TitleText,
                (
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(1.00 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_InlayText,
                (
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(1.00 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_XAxis,
                (
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(1.00 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_XAxisGrid,
                (
                    int(1.00 * 255),
                    int(1.00 * 255),
                    int(1.00 * 255),
                    int(1.00 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_YAxis,
                (
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(1.00 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_YAxisGrid,
                (
                    int(1.00 * 255),
                    int(1.00 * 255),
                    int(1.00 * 255),
                    int(1.00 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_YAxis2,
                (
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(1.00 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_YAxisGrid2,
                (
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(0.50 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_YAxis3,
                (
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(1.00 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_YAxisGrid3,
                (
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(0.50 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Selection,
                (
                    int(0.82 * 255),
                    int(0.64 * 255),
                    int(0.03 * 255),
                    int(1.00 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Query,
                (
                    int(int(0.00 * 255)),
                    int(0.84 * 255),
                    int(0.37 * 255),
                    int(1.00 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Crosshairs,
                (
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(int(0.00 * 255)),
                    int(0.50 * 255),
                ),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_NodeBackground,
                (240, 240, 240, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_NodeBackgroundHovered,
                (240, 240, 240, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_NodeBackgroundSelected,
                (240, 240, 240, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_NodeOutline,
                (100, 100, 100, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_TitleBar,
                (248, 248, 248, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_TitleBarHovered,
                (209, 209, 209, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_TitleBarSelected,
                (209, 209, 209, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_Link, (66, 150, 250, 100), category=dpg.mvThemeCat_Nodes
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_LinkHovered,
                (66, 150, 250, 242),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_LinkSelected,
                (66, 150, 250, 242),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_Pin, (66, 150, 250, 160), category=dpg.mvThemeCat_Nodes
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_PinHovered,
                (66, 150, 250, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_BoxSelector,
                (90, 170, 250, 30),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_BoxSelectorOutline,
                (90, 170, 250, 150),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_GridBackground,
                (225, 225, 225, 255),
                category=dpg.mvThemeCat_Nodes,
            )
            dpg.add_theme_color(
                dpg.mvNodeCol_GridLine,
                (180, 180, 180, 100),
                category=dpg.mvThemeCat_Nodes,
            )

    return theme_id  # type: ignore
