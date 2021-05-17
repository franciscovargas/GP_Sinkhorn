from gp_sinkhorn.utils import get_subject_data, animate_skeleton


if __name__ == "__main__":
    subject, motion = "35", ["01"]

    data = get_subject_data(subject=subject, motion=motion)

    Y = data["Y"]

    camera = animate_skeleton(Y, data, standardise=True, notebook=False)
    animation = camera.animate()
    print("done animation now saving")
    animation.save('my_animation_standardised_equal.mp4')
    print("done saving animation")
