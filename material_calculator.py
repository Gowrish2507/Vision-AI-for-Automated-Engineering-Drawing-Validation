import math

def calculate_global_materials(wall_length, wall_height, column_qty):
    """Calculates Bricks and Concrete."""
    wall_volume = wall_length * wall_height * 0.23 # 9-inch wall
    num_bricks = int(wall_volume / 0.002)
    concrete_volume = column_qty * (0.3 * 0.3 * wall_height)
    return num_bricks, concrete_volume

def calculate_room_materials(room_area, wall_height):
    """Calculates Tiles, Paint, and Cement for a specific room."""
    if room_area <= 0:
        return 0, 0.0, 0
        
    # Tiles (600x600mm + 5% wastage)
    num_tiles = math.ceil((room_area / 0.36) * 1.05)
    
    # Paint (Perimeter * Height / 10 * 2 coats)
    perimeter = 4 * math.sqrt(room_area)
    paint_liters = round(((perimeter * wall_height) / 10) * 2, 1)
    
    # Cement (Ceiling Plaster 12mm)
    cement_bags = math.ceil((room_area * 0.012) * 8)
    
    return num_tiles, paint_liters, cement_bags