settings = {
    'intr_area_x': 100,
    'intr_area_y': 100,
    'step_x': 0.5,
    'step_y': 0.5,
    'scan_area_ratio_x': 0.5,
    'scan_area_ratio_y': 0.5
}

settings['border_x'] = int(settings['intr_area_x'] * settings['scan_area_ratio_x'])
settings['border_y'] = int(settings['intr_area_y'] * settings['scan_area_ratio_y'])
