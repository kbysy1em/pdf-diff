settings = {
    'intr_area_x': 100,
    'intr_area_y': 100,
    'step_x': 0.5,
    'step_y': 0.5,
    'scan_area_ratio_x': 0.5,
    'scan_area_ratio_y': 0.5,
    'criterion': 0.98,  # 0.98

    # color
    'green': (0, 128, 0),
    'pink': (203, 192, 255),
    'red' : (0, 0, 255),
    'blue' : (255, 0, 0),

    # others
    'debug': False
}

settings['border_x'] = int(settings['intr_area_x'] * settings['scan_area_ratio_x'])
settings['border_y'] = int(settings['intr_area_y'] * settings['scan_area_ratio_y'])
