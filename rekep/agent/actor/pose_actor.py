class PoseActor(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.grounded_dino = GroundedDINO(config)
        self.vlm = VLMInterface(config)
        
    def generate_pose(self, observation, instruction):
        """Generate 6DOF pose with semantic understanding"""
        # 1. Get semantic keypoints
        keypoints = self.grounded_dino.detect(observation)
        
        # 2. Generate pose constraints
        constraints = self.vlm.generate_constraints(
            observation, 
            instruction,
            keypoints
        )
        
        # 3. Optimize pose
        return self.optimize_pose(keypoints, constraints)