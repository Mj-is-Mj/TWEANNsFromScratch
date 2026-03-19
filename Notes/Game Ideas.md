## Desired Elements

### Environment
**Partially Observable**, but close to fully observable. Should be "tidbits" of invisible information, e.g. there should be
**Single-agent**
**Deterministic**
**Sequential**
**Dynamic**, but only slightly. I want there to be some concept of "thinking time" so the agent can choose to wait and have the environment continue on anyways
**Continuous**, with notable discrete elements (e.g. boolean states)
**Unknown**

### Other details
**Computer vision:** At least one game/task without any computer vision, but one with computer vision would be good as well
**Stochasticity:** The environments should be generated stochastically, but state-space navigation will be deterministic. 

# Specific Ideas

## Iron Lung
Can be made infinite using

### AI-Related Problems/Solutions
Different size curr_positions and goal_positions might cause confusion
	Instead of using raw positions, use only relative positions, i.e. the only position the AI can read is *goal - curr* 
Discontinuity in angle might lead to cyclical behaviour (e.g. it wants angle < 0, but it wraps back to 360 so it just spins)
	Use a normalized vector to represent angle input
