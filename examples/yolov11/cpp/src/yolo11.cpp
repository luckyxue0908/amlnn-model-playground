#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <float.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "nn_sdk.h"
#include <cstdio> 
#include <numeric>

namespace fs = std::filesystem;

static constexpr int kInputW = 640;
static constexpr int kInputH = 640;
static constexpr int kNumClasses = 80;
static constexpr int kDflChannels = 16;
static constexpr int kTotalChannels = 144; 

static const std::vector<std::string> kClassNames = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

static void hwc_to_chw(const cv::Mat& src, float* dst) {
    int h = src.rows, w = src.cols;
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                dst[k * h * w + i * w + j] = src.at<cv::Vec3f>(i, j)[k];
            }
        }
    }
}

static float decode_dfl(const float* dfl_ptr) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < kDflChannels; i++) {
        if (dfl_ptr[i] > max_val) max_val = dfl_ptr[i];
    }

    float exp_sum = 0;
    float exp_vals[kDflChannels];
    for (int i = 0; i < kDflChannels; i++) {
        exp_vals[i] = std::exp(dfl_ptr[i] - max_val);
        exp_sum += exp_vals[i];
    }

    float res = 0;
    for (int i = 0; i < kDflChannels; i++) {
        res += (exp_vals[i] / exp_sum) * i;
    }
    return res;
}

static float calculate_iou(const cv::Rect& a, const cv::Rect& b) {
    int xx1 = std::max(a.x, b.x);
    int yy1 = std::max(a.y, b.y);
    int xx2 = std::min(a.x + a.width, b.x + b.width);
    int yy2 = std::min(a.y + a.height, b.y + b.height);

    int w = std::max(0, xx2 - xx1);
    int h = std::max(0, yy2 - yy1);
    float inter = (float)w * h;
    float areaA = (float)a.width * a.height;
    float areaB = (float)b.width * b.height;
    return inter / (areaA + areaB - inter);
}

static std::vector<int> manual_nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float thresh) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { 
        return scores[a] > scores[b]; 
    });

    std::vector<int> keep;
    while (!order.empty()) {
        int i = order[0];
        keep.push_back(i);
        order.erase(order.begin());
        
        order.erase(std::remove_if(order.begin(), order.end(), [&](int j) { 
            return calculate_iou(boxes[i], boxes[j]) > thresh; 
        }), order.end());
    }
    return keep;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir>\n";
        return 0;
    }

    aml_config cfg{};
    cfg.typeSize = sizeof(cfg);
    cfg.modelType = ADLA_LOADABLE;
    cfg.nbgType = NN_ADLA_FILE;
    cfg.path = argv[1];
    void* ctx = aml_module_create(&cfg);
    if (!ctx) {
        std::cerr << "Failed to create aml module\n";
        return -1;
    }

    std::vector<float> chw_buffer(kInputW * kInputH * 3);
    fs::create_directory("yolo11_result");

    for (auto& it : fs::directory_iterator(argv[2])) {
        cv::Mat img = cv::imread(it.path().string());
        if (img.empty()) continue;

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing image: " << it.path().filename() << std::endl;
        std::cout << "============================================================" << std::endl;

        float scale = std::min((float)kInputW / img.cols, (float)kInputH / img.rows);
        int nw = img.cols * scale, nh = img.rows * scale;
        int px = (kInputW - nw) / 2, py = (kInputH - nh) / 2;
        
        cv::Mat res, canvas = cv::Mat::zeros(kInputH, kInputW, CV_32FC3);
        canvas.setTo(cv::Scalar(114.0/255.0, 114.0/255.0, 114.0/255.0)); 
        cv::resize(img, res, {nw, nh});
        res.convertTo(res, CV_32FC3, 1.0 / 255.0); 
        res.copyTo(canvas(cv::Rect(px, py, nw, nh)));
        
        hwc_to_chw(canvas, chw_buffer.data());

        nn_input in{};
        in.typeSize = sizeof(in);
        in.input_type = BINARY_RAW_DATA;
        in.input = (unsigned char*)chw_buffer.data();
        in.size = chw_buffer.size() * sizeof(float);
        in.info.valid = 1;
        in.info.input_format = AML_INPUT_MODEL_NCHW;
        in.info.input_data_type = AML_INPUT_FP32;
        aml_module_input_set(ctx, &in);

        aml_output_config_t outcfg{};
        outcfg.typeSize = sizeof(outcfg);
        outcfg.format = AML_OUTDATA_FLOAT32;
        nn_output* out = (nn_output*)aml_module_output_get(ctx, outcfg);
        if (!out) continue;

        std::cout << "Output count: " << out->num << std::endl;

        std::vector<cv::Rect> bboxes;
        std::vector<float> confs;
        std::vector<int> class_ids;
        std::vector<int> strides = {32, 16, 8}; 

        for (int i = 0; i < out->num; i++) {
            float* data = (float*)out->out[i].buf;
            int stride = strides[i];
            int grid_h = kInputH / stride;
            int grid_w = kInputW / stride;
            int num_elements = out->out[i].size / sizeof(float);

            float min_v = FLT_MAX, max_v = -FLT_MAX, sum_v = 0;
            for (int j = 0; j < num_elements; j++) {
                if (data[j] < min_v) min_v = data[j];
                if (data[j] > max_v) max_v = data[j];
                sum_v += data[j];
            }
            printf("Output[%d] shape=(%d,%d,%d) min=%.6f max=%.6f mean=%.6f\n", 
                   i, grid_h, grid_w, kTotalChannels, min_v, max_v, sum_v / num_elements);

            for (int g = 0; g < grid_h * grid_w; g++) {
                float* feat = data + g * kTotalChannels;
                
                float max_score = -1.0f;
                int cls_id = -1;
                for (int c = 0; c < kNumClasses; c++) {
                    float score = 1.0f / (1.0f + std::exp(-feat[64 + c]));
                    if (score > max_score) {
                        max_score = score;
                        cls_id = c;
                    }
                }

                if (max_score > 0.3f) { 
                    float d_l = decode_dfl(feat + 0);
                    float d_t = decode_dfl(feat + 16);
                    float d_r = decode_dfl(feat + 32);
                    float d_b = decode_dfl(feat + 48);

                    float cx = ((g % grid_w) + 0.5f);
                    float cy = ((g / grid_w) + 0.5f);

                    float x1 = (cx - d_l) * stride;
                    float y1 = (cy - d_t) * stride;
                    float x2 = (cx + d_r) * stride;
                    float y2 = (cy + d_b) * stride;

                    int rx1 = std::max(0, (int)((x1 - px) / scale));
                    int ry1 = std::max(0, (int)((y1 - py) / scale));
                    int rx2 = std::min(img.cols, (int)((x2 - px) / scale));
                    int ry2 = std::min(img.rows, (int)((y2 - py) / scale));

                    bboxes.push_back(cv::Rect(rx1, ry1, rx2 - rx1, ry2 - ry1));
                    confs.push_back(max_score);
                    class_ids.push_back(cls_id);
                }
            }
        }

        std::vector<int> indices = manual_nms(bboxes, confs, 0.45f);
        
        if (!indices.empty()) {
            printf("    Detected %d objects:\n", (int)indices.size());
            for (size_t i = 0; i < indices.size(); i++) {
                int idx = indices[i];
                std::string label = kClassNames[class_ids[idx]];
                printf("      %zu. %s (%.2f)\n", i + 1, label.c_str(), confs[idx]);

                cv::rectangle(img, bboxes[idx], {0, 255, 0}, 2);
                char text[256];
                std::sprintf(text, "%s %.2f", label.c_str(), confs[idx]);
                cv::putText(img, text, {bboxes[idx].x, bboxes[idx].y - 5}, 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
            }
        } else {
            printf("    No objects detected\n");
        }

        cv::imwrite("yolo11_result/" + it.path().filename().string(), img);
        std::cout << "Result saved to: yolo11_result/" << it.path().filename() << "\n\n";
    }

    aml_module_destroy(ctx);
    return 0;
}