settings = {
    # comparison settings
    'intr_area_x': 100,
    'intr_area_y': 100,
    'step_x': 0.5,
    'step_y': 0.5,
    'scan_area_ratio_x': 0.5,
    'scan_area_ratio_y': 0.5,
    'criterion': 3.9,

    # input file info
    'rotate1': '',  # 'cw, ccw or empty
    'rotate2': '',

    # color
    'cv_pink': (203, 192, 255),
    'pink': (255, 192, 203),
    'cv_red' : (0, 0, 255),
    'red': (255, 0, 0),
    'deep_green': (0, 128, 0),
    'cv_green': (0, 128, 0),
    'blue' : (0, 0, 255),

    # others
    'inverse_comparison' : 'left',  # 'left, right or empty'
    'debug': False
}

settings['border_x'] = int(settings['intr_area_x'] * settings['scan_area_ratio_x'])
settings['border_y'] = int(settings['intr_area_y'] * settings['scan_area_ratio_y'])
settings['cv_blue'] = tuple(reversed(settings['blue']))
