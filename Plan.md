1. Do detection on a video:
    - Requires ball tracking -- good enough
    - Player foot tracking -- kinda done
    - Court detection -- good enough

    ** need to test all of these baselines, but can do so after pipeline is made

2. Combine detections of ball and court to get 3d Direct linear transform
    - Need a way to evaluate this...eye test?

3. Create a basic optimization problem - newtons equations for trajectory estimation
    - Need a way to segment the shots
        - Test change in direction with a homography transform?
        - Test label wall hitting with some sort of detection network?
        - Something else? Can discuss with Jihoon or olga

4. Test against detections:
    - Need:
        1. To write metric testing (IOU, precision stuff?)
        2. I feel like something is missing...

5. Refine using the labeled data
- retraining detectors, refining hit detection, better shot optimization, etc.

Most important for a draft paper?
- 
Definitely have pipeline down, show that the concept is feasible - points 1, 2, 3

Would be good to have baseline models without finetuning, so metrics are still important

Then finally can test against labeled data, but that is least important now - finetuning comes later

