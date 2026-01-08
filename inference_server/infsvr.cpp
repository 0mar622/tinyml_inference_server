#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // -------- 1. Create runtime environment --------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "infer");

    // -------- 2. Session options (CPU) --------
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // -------- 3. Load ONNX model --------
    const char* model_path = "../models/mobilenetv2_cifar10.onnx";
    Ort::Session session(env, model_path, session_options);

    // -------- 4. Allocator --------
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name  = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);

    std::cout << "Input name: " << input_name.get() << std::endl;
    std::cout << "Output name: " << output_name.get() << std::endl;

    // -------- 5. Load & preprocess image --------
    cv::Mat img = cv::imread("../client/dog.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load image\n";
        return 1;
    }

    cv::resize(img, img, cv::Size(224, 224));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // NCHW tensor
    std::vector<float> input_tensor_values(1 * 3 * 224 * 224);
    int idx = 0;

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                float val = img.at<cv::Vec3f>(y, x)[c];
                float mean = (c == 0) ? 0.485f : (c == 1 ? 0.456f : 0.406f);
                float std  = (c == 0) ? 0.229f : (c == 1 ? 0.224f : 0.225f);
                input_tensor_values[idx++] = (val - mean) / std;
            }
        }
    }

    // -------- 6. Create input tensor --------
    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // -------- 7. Run inference --------
    const char* input_names[]  = { input_name.get() };
    const char* output_names[] = { output_name.get() };

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    // -------- 8. Read output --------
    float* scores = output_tensors[0].GetTensorMutableData<float>();

    int best_class = 0;
    float best_score = scores[0];
    for (int i = 1; i < 10; i++) {
        if (scores[i] > best_score) {
            best_score = scores[i];
            best_class = i;
        }
    }

    std::cout << "Predicted class: " << best_class
              << "  logit: " << best_score << std::endl;

    return 0;
}
