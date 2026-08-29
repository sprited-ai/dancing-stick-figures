import numpy as np

from eval.oracle import NAMES, PAL, label_colours, score_frame


def canvas():
    image = np.zeros((32, 32, 4), np.uint8)
    image[20:25, 20:25, :3] = (150, 150, 150)
    image[20:25, 20:25, 3] = 255
    return image


def paint(image, points, colour):
    for y, x in points:
        image[y, x, :3] = colour
        image[y, x, 3] = 255


def test_four_pixels_are_present_but_three_are_absent():
    three = canvas()
    paint(three, [(5, 5), (5, 6), (6, 5)], PAL["arm_L"])
    four = canvas()
    paint(four, [(5, 5), (5, 6), (6, 5), (6, 6)], PAL["arm_L"])
    assert score_frame(three)["ncomp"]["arm_L"] == 0
    assert score_frame(four)["ncomp"]["arm_L"] == 1


def test_diagonal_contact_is_one_eight_connected_component():
    image = canvas()
    paint(image, [(5, 5), (5, 6), (6, 7), (7, 7)], PAL["arm_L"])
    assert score_frame(image)["ncomp"]["arm_L"] == 1


def adjacency_frame(distance):
    image = canvas()
    paint(image, [(5, 5), (5, 6), (6, 5), (6, 6)], PAL["ink"])
    start = 6 + distance
    paint(image, [(5, start), (5, start + 1), (6, start), (6, start + 1)], PAL["arm_L"])
    return image


def test_lie_allows_two_pixel_chebyshev_distance_but_not_three():
    at_two = score_frame(adjacency_frame(2))["lie"]
    at_three = score_frame(adjacency_frame(3))["lie"]
    assert at_two == 7 / 8
    assert at_three == 1.0
