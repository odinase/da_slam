#ifndef VISUALIZER_H
#define VISUALIZER_H


class Visualizer {
    public:
    VisualizerNode(const ros::NodeHandle& nh);
    private:
    ros::NodeHandle nh_;

};


#endif // VISUALIZER_H