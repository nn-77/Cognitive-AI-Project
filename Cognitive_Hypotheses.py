"""
    Functions to implement cognition-based rules for the Berzerk Atari game.
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import math


PLAYER_COLOR = np.array([240, 170, 103])
WALL_COLOR = np.array([84, 92, 214])
PLAYER_MAP_GRAYSCALE = 3
WALL_MAP_GRAYSCALE = 1

# the custom action space used is CUSTOM_ACTION_SPACE = [1, 2, 3, 4, 5]
# 1 is fire, 2 is up, etc..., and 1 is in array spot 0, 2 is in array spot 1, etc.
FIRE_INDEX = 0
UP_INDEX = 1
RIGHT_INDEX = 2
LEFT_INDEX = 3
DOWN_INDEX = 4

TOLERANCE = 7


def cognitive_wrapper(logits, rgb_pixels, metadata):
    """
        Apply the cognition-based rules to the given logits based on the current screen state (rgb_pixels)
        and return an appropriate action, sampling from the modified logits.

        $1: Alter the bounds of the "tight" bounding box found in the function above. Note that a tighter bounding box
            is better than a looser one here because we can shoot rays from the box towards other objects to be
            "overcorrective" (better than undercorrective)
                - tradeoff between expanding the bounding box vs not
    """

    handle_metadata(metadata)

    simplified_map, only_player_and_bullets_map = simplify_screen(rgb_pixels)

    bounding_coords = find_bounding_coords_of_player(only_player_and_bullets_map)

    # if player not found, that means this is an irrelevant frame (cutscene, or similar)
    if bounding_coords is None:
        return np.random.randint(logits.shape[1])

    # $1
    bounding_coords = find_bounding_box(simplified_map, bounding_coords)

    # apply the cognitive elements based on the simplified view of the game and the bounding box obtained
    left_right_top_bot_closest_walls = find_closest_walls_all_directions(simplified_map, bounding_coords)

    # adjust the logits (viable actions) based on the collected information
    adjust_bias_directions(left_right_top_bot_closest_walls, metadata)
    adjust_logits_based_on_bias(logits, metadata)
    adjust_logits_for_wall_dist(logits, left_right_top_bot_closest_walls)

    logits_softmax = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(logits_softmax, 1).item()


def handle_metadata(metadata):
    if "horizontal_bias" not in metadata:
        metadata["horizontal_bias"] = "right" if np.random.random() < 0.5 else "left"
    if "vertical_bias" not in metadata:
        metadata["vertical_bias"] = "top" if np.random.random() < 0.5 else "bottom"


def simplify_screen(rgb_pixels):
    """
        Simplifies the image to the components we care about - the player pixels and the walls. The bullets are
        the same color as the player, so we need to return the min/max x/y values surrounding the torso of the player,
        to generate a bounding box later

        - raw image is HxWx3
    """

    """
        Old code that was causing extreme slowdown - left for reference, replaced by the below equivalent
        vectorized numpy operations
    """
                # simplified_map = np.zeros((rgb_pixels.shape[0], rgb_pixels.shape[1]))
                # only_player_and_bullets_map = np.zeros((rgb_pixels.shape[0], rgb_pixels.shape[1]))
                # for i in range(rgb_pixels.shape[0]):
                #     for j in range(rgb_pixels.shape[1]):
                #         if (PLAYER_COLOR[0] - 7 <= rgb_pixels[i][j][0] <= PLAYER_COLOR[0] + 7
                #                 and PLAYER_COLOR[1] - 7 <= rgb_pixels[i][j][1] <= PLAYER_COLOR[1] + 7
                #                 and PLAYER_COLOR[2] - 7 <= rgb_pixels[i][j][2] <= PLAYER_COLOR[2] + 7):
                #             simplified_map[i][j] = PLAYER_MAP_GRAYSCALE
                #             only_player_and_bullets_map[i][j] = 1
                #
                #         elif (WALL_COLOR[0] - 7 <= rgb_pixels[i][j][0] <= WALL_COLOR[0] + 7
                #                 and WALL_COLOR[1] - 7 <= rgb_pixels[i][j][1] <= WALL_COLOR[1] + 7
                #                 and WALL_COLOR[2] - 7 <= rgb_pixels[i][j][2] <= WALL_COLOR[2] + 7):
                #             simplified_map[i][j] = WALL_MAP_GRAYSCALE

    # this is the simplified screen that is returned
    simplified_map = np.zeros_like(rgb_pixels[..., 0], dtype=np.uint8)

    # this is more of a temp map used to find the bounding box around the player
    only_player_and_bullets_map = np.zeros_like(rgb_pixels[..., 0], dtype=np.uint8)

    player_mask = np.all((rgb_pixels >= PLAYER_COLOR - 7) & (rgb_pixels <= PLAYER_COLOR + 7), axis=-1)
    wall_mask = np.all((rgb_pixels >= WALL_COLOR - 7) & (rgb_pixels <= WALL_COLOR + 7), axis=-1)

    simplified_map[player_mask] = PLAYER_MAP_GRAYSCALE
    simplified_map[wall_mask] = WALL_MAP_GRAYSCALE
    only_player_and_bullets_map[player_mask] = 1

    # uncomment to show side by side of raw image vs simplified map (no bullets removed)
    # demo_show_images_simplified_map(rgb_pixels, simplified_map)

    return simplified_map, only_player_and_bullets_map


def find_bounding_coords_of_player(only_player_and_bullets_map):
    """
        Connected components to find the min x, min y, max x, and max y of the player
        
        - Arms and head and legs and torso and bullets are all the same color (and separate components due to gaps), but
          one thing stays consistent - the torso is always the biggest connected component. So it can be used to
          generate a sufficient bounding box; we just need the boundary coords of the torso
    """
    only_player_and_bullets_map, num_features = label(only_player_and_bullets_map)

    # uncomment to show side by side of raw image vs segmented groups of pixels that have the same color as the player
    # demo_show_segmented(rgb_pixels, only_player_and_bullets_map, num_features)

    mx = 0
    mx_connected_component = []
    for component in range(1, num_features + 1):
        coords = np.argwhere(only_player_and_bullets_map == component)
        if len(coords) > mx:
            mx_connected_component = coords
            mx = len(coords)

    # if player not found, that means this is an irrelevant frame (cutscene, or similar)
    if len(mx_connected_component) == 0:
        return None

    maxX = mx_connected_component[0][1]
    maxY = mx_connected_component[0][0]
    minX = mx_connected_component[0][1]
    minY = mx_connected_component[0][0]
    for coord in mx_connected_component:
        maxX = max(maxX, coord[1])
        maxY = max(maxY, coord[0])
        minX = min(minX, coord[1])
        minY = min(minY, coord[0])

    return [minY, minX, maxY, maxX]


def find_bounding_box(simplified_map, bounding_coords):
    """
        Adjust the bounding box based on empirical / realistic scenarios
    """

    new_bounding_coords = [bounding_coords[0] - 7, bounding_coords[1] - 0, bounding_coords[2] + 0, bounding_coords[3] + 0]

    # uncomment to show the simplified image with the found bounding box
    # demo_found_bounding_box(simplified_map, new_bounding_coords)

    return new_bounding_coords


def find_closest_walls_all_directions(simplified_map, bounding_coords):
    """
        Return the closest wall from the left, right, top, and bottom of the player
    """

    minY, minX, maxY, maxX = bounding_coords

    screen_min_y = 0
    screen_min_x = 0
    screen_max_y = simplified_map.shape[0]
    screen_max_x = simplified_map.shape[1]

    # 3 rays shot from each side of the bounding box so that walls of all sizes at all locations relative to the player
    # are considered (of course 3 is not fine-grained enough to cover ALL cases, but this estimation works well enough)
        # to cover the case, where for example, the xxxx are the player torso and the //// are the walls:
        #    xx  --------->////
        #    xxx ---->////////
        #    xx  --------->//

        # or this case:
        #    xx  ---->////////
        #    xxx --------->///
        #    xx  --------->//

        # or this case:
        #    xx  ----------->/
        #    xxx --------->///
        #    xx  --->/////////

    top_ray_1 = (minY, minX)
    top_ray_2 = (minY, (minX + maxX) // 2)
    top_ray_3 = (minY, maxX)
    top_closest_dist = math.inf
    for y_val in range(top_ray_1[0], screen_min_y, -1):
        # move the rays
        top_ray_1 = (y_val, top_ray_1[1])
        top_ray_2 = (y_val, top_ray_2[1])
        top_ray_3 = (y_val, top_ray_3[1])

        # check if any of the rays intersect a wall; if so, break since we only want the closest wall
        if (simplified_map[top_ray_1[0]][top_ray_1[1]] == WALL_MAP_GRAYSCALE
                or simplified_map[top_ray_2[0]][top_ray_2[1]] == WALL_MAP_GRAYSCALE
                or simplified_map[top_ray_3[0]][top_ray_3[1]] == WALL_MAP_GRAYSCALE):
            top_closest_dist = minY - y_val
            break

    bottom_ray_1 = (maxY, minX)
    bottom_ray_2 = (maxY, (minX + maxX) // 2)
    bottom_ray_3 = (maxY, maxX)
    bottom_closest_dist = math.inf
    for y_val in range(bottom_ray_1[0], screen_max_y):
        # move the rays
        bottom_ray_1 = (y_val, bottom_ray_1[1])
        bottom_ray_2 = (y_val, bottom_ray_2[1])
        bottom_ray_3 = (y_val, bottom_ray_3[1])

        # check if any of the rays intersect a wall; if so, break since we only want the closest wall
        if (simplified_map[bottom_ray_1[0]][bottom_ray_1[1]] == WALL_MAP_GRAYSCALE
                or simplified_map[bottom_ray_2[0]][bottom_ray_2[1]] == WALL_MAP_GRAYSCALE
                or simplified_map[bottom_ray_3[0]][bottom_ray_3[1]] == WALL_MAP_GRAYSCALE):
            bottom_closest_dist = y_val - maxY
            break

    left_ray_1 = (minY, minX)
    left_ray_2 = ((minY + maxY) // 2, minX)
    left_ray_3 = (maxY, minX)
    left_closest_dist = math.inf
    for x_val in range(left_ray_1[1], screen_min_x, -1):
        # move the rays
        left_ray_1 = (left_ray_1[0], x_val)
        left_ray_2 = (left_ray_2[0], x_val)
        left_ray_3 = (left_ray_3[0], x_val)

        # check if any of the rays intersect a wall; if so, break since we only want the closest wall
        if (simplified_map[left_ray_1[0]][left_ray_1[1]] == WALL_MAP_GRAYSCALE
                or simplified_map[left_ray_2[0]][left_ray_2[1]] == WALL_MAP_GRAYSCALE
                or simplified_map[left_ray_3[0]][left_ray_3[1]] == WALL_MAP_GRAYSCALE):
            left_closest_dist = minX - x_val
            break

    right_ray_1 = (minY, maxX)
    right_ray_2 = ((minY + maxY) // 2, maxX)
    right_ray_3 = (maxY, maxX)
    right_closest_dist = math.inf
    for x_val in range(right_ray_1[1], screen_max_x):
        # move the rays
        right_ray_1 = (right_ray_1[0], x_val)
        right_ray_2 = (right_ray_2[0], x_val)
        right_ray_3 = (right_ray_3[0], x_val)

        # check if any of the rays intersect a wall; if so, break since we only want the closest wall
        if (simplified_map[right_ray_1[0]][right_ray_1[1]] == WALL_MAP_GRAYSCALE
                or simplified_map[right_ray_2[0]][right_ray_2[1]] == WALL_MAP_GRAYSCALE
                or simplified_map[right_ray_3[0]][right_ray_3[1]] == WALL_MAP_GRAYSCALE):
            right_closest_dist = x_val - maxX
            break

    left_right_top_bot_closest_walls = (left_closest_dist, right_closest_dist, top_closest_dist, bottom_closest_dist)

    # uncomment to show images of the simplified scene with the found closest walls
    # demo_found_closest_walls(simplified_map, bounding_coords, left_right_top_bot_closest_walls)

    return left_right_top_bot_closest_walls


def adjust_bias_directions(left_right_top_bot_closest_walls, metadata):
    if metadata["horizontal_bias"] == "left" and left_right_top_bot_closest_walls[0] <= TOLERANCE:
        metadata["horizontal_bias"] = "right"
    elif metadata["horizontal_bias"] == "right" and left_right_top_bot_closest_walls[1] <= TOLERANCE:
        metadata["horizontal_bias"] = "left"

    if metadata["vertical_bias"] == "top" and left_right_top_bot_closest_walls[2] <= TOLERANCE:
        metadata["vertical_bias"] = "bottom"
    elif metadata["vertical_bias"] == "bottom" and left_right_top_bot_closest_walls[3] <= TOLERANCE:
        metadata["vertical_bias"] = "top"


def adjust_logits_based_on_bias(logits, metadata):
    # scale the appropriate ones down (instead of scaling their counterparts up) so that the shooting probability isnt affected
    if metadata["horizontal_bias"] == "left":
        logits[0, RIGHT_INDEX] -= 8 * abs(logits[0, RIGHT_INDEX])
    elif metadata["horizontal_bias"] == "right":
        logits[0, LEFT_INDEX] -= 8 * abs(logits[0, LEFT_INDEX])

    if metadata["vertical_bias"] == "top":
        logits[0, DOWN_INDEX] -= 8 * abs(logits[0, DOWN_INDEX])
    elif metadata["vertical_bias"] == "bottom":
        logits[0, UP_INDEX] -= 8 * abs(logits[0, UP_INDEX])


def adjust_logits_for_wall_dist(logits, left_right_top_bot_closest_walls):
    if left_right_top_bot_closest_walls[0] <= TOLERANCE:
        logits[0, LEFT_INDEX] = -math.inf
    if left_right_top_bot_closest_walls[1] <= TOLERANCE:
        logits[0, RIGHT_INDEX] = -math.inf
    if left_right_top_bot_closest_walls[2] <= TOLERANCE:
        logits[0, UP_INDEX] = -math.inf
    if left_right_top_bot_closest_walls[3] <= TOLERANCE:
        logits[0, DOWN_INDEX] = -math.inf


###################################################################################################


def demo_show_images_simplified_map(original, modified):
    print("!!!!!!!!!!!!!!!\nSHOWING IMAGES\n!!!!!!!!!!!!!!!!")
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(original)
    axes[1].imshow(modified, cmap='gray')
    plt.show()


def demo_show_segmented(original, modified, nf):
    print("!!!!!!!!!!!!!!!\nSHOWING IMAGES\n!!!!!!!!!!!!!!!!")
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(original)
    axes[1].imshow(modified, cmap='gray')
    axes[1].set_title(f"# features: {nf}")
    plt.show()


def demo_found_bounding_box(simplified_map, bounding_coords):
    print("!!!!!!!!!!!!!!!\nSHOWING IMAGES\n!!!!!!!!!!!!!!!!")
    cpy = np.copy(simplified_map)
    minY, minX, maxY, maxX = bounding_coords
    for i in range(minY, maxY):
        cpy[i, minX] = 20
        cpy[i, maxX] = 20
    for i in range(minX, maxX):
        cpy[minY, i] = 20
        cpy[maxY, i] = 20

    plt.imshow(cpy)
    plt.show()


def demo_found_closest_walls(simplified_map, bounding_coords, left_right_top_bot_closest_walls):
    print("!!!!!!!!!!!!!!!\nSHOWING IMAGES\n!!!!!!!!!!!!!!!!")
    cpy = np.copy(simplified_map)
    minY, minX, maxY, maxX = bounding_coords
    for i in range(minY, maxY):
        cpy[i, minX] = 20
        cpy[i, maxX] = 20
    for i in range(minX, maxX):
        cpy[minY, i] = 20
        cpy[maxY, i] = 20

    plt.imshow(cpy)
    plt.title(f"left: {left_right_top_bot_closest_walls[0]}; right: {left_right_top_bot_closest_walls[1]};"
              f" top: {left_right_top_bot_closest_walls[2]}; bottom: {left_right_top_bot_closest_walls[3]}")
    plt.show()
