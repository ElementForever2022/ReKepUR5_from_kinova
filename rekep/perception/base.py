class PerceptionPipeline:
    def __init__(self):
        self.fusion = MultiViewFusion()
        self.semantic = GroundedDINO()
        
    def process(self, multi_view_frames):
        # 1. 多视角融合
        point_cloud = self.fusion.fuse_point_clouds(multi_view_frames)
        
        # 2. 语义理解
        semantic_keypoints = self.semantic.detect(point_cloud)
        
        return semantic_keypoints
    


if __name__ == "__main__":
    perception = PerceptionPipeline()
    semantic_keypoints = perception.process(multi_view_frames)
    print(semantic_keypoints)