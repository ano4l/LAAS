"""Scoring the chosen ball against labelled balls: every labelled frame goes in exactly one bin."""
from workers.video.ball_benchmark import load_balls, merge, score


def ball(frame, x, y, chosen=False, confidence=.6, **extra):
    return dict(frame_idx=frame, object_type="ball", bbox=[x - 5, y - 5, x + 5, y + 5], confidence=confidence, ball_selected=chosen, model_role="object_detector", **extra)


def test_labels_give_the_ball_box_and_the_people_in_each_frame():
    doc = dict(images=[dict(image_id="a", file_name="000001.jpg"), dict(image_id="b", file_name="000002.jpg")], annotations=[
        dict(image_id="a", supercategory="object", attributes=dict(role="ball"), bbox_image=dict(x=100, y=200, w=10, h=10)),
        dict(image_id="b", supercategory="object", attributes=dict(role="player"), bbox_image=dict(x=50, y=60, w=20, h=60)),
        dict(image_id="b", supercategory="pitch")])
    balls, people = load_balls(doc)
    assert balls == {0: (100, 200, 110, 210)} and people == {1: [(50, 60, 70, 120)]}


def test_each_labelled_frame_is_right_wrong_detected_but_not_chosen_or_not_detected():
    balls = {0: (95, 95, 105, 105), 1: (115, 95, 125, 105), 2: (135, 95, 145, 105), 3: (155, 95, 165, 105)}
    rows = [ball(0, 101, 99, chosen=True),                             # the chosen ball is the labelled one
            ball(1, 120, 100), ball(1, 600, 300, chosen=True),          # the ball was found, but something else was chosen
            ball(2, 141, 101),                                          # found and nothing chosen
            ball(4, 400, 400, chosen=True),                             # chosen in a frame with no labelled ball
            ball(3, 160, 100, chosen=True, ball_rejected="painted_spot")]   # a rejected candidate does not count
    report = score(rows, balls, {2: [(130, 60, 150, 120)]}, frames=5)
    assert report["bins"] == dict(right=1, wrong=1, detected_not_chosen=1, not_detected=1)
    assert report["detected_frames"] == 3 and report["chosen_frames"] == 3 and report["chosen_without_a_labelled_ball"] == 1
    assert report["by"]["place"]["at a player's body in the image"] == dict(right=0, wrong=0, detected_not_chosen=1, not_detected=0)
    assert report["by"]["speed"]["20 px a frame and over"]["wrong"] == 1 and report["found_by"] == dict(object_detector=3)
    both = merge([report, report])
    assert both["labelled_frames"] == 8 and both["recall_chosen"] == .25 and both["recall_detected"] == .75 and both["precision_chosen"] == round(2 / 6, 3)
    assert both["by"]["size"]["8 to 12 px"]["not_detected"] == .25
