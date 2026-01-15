#include "httplib.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ---------------- Labels ----------------
static const char* CIFAR10_LABELS[10] = {
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
};

// ---------------- Runtime globals ----------------
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "server");
Ort::SessionOptions session_options;
Ort::Session* session_ptr = nullptr;
Ort::AllocatorWithDefaultOptions allocator;

std::string input_name;
std::string output_name;

// ---------------- Softmax ----------------
std::vector<float> softmax(const float* logits, int n) {
    float maxv = logits[0];
    for (int i = 1; i < n; i++)
        maxv = std::max(maxv, logits[i]);

    std::vector<float> probs(n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = std::exp(logits[i] - maxv);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++)
        probs[i] /= sum;

    return probs;
}

// ---------------- Inference ----------------
int run_inference_from_mat(cv::Mat img, float& confidence) {
    if (img.empty()) return -1;

    cv::resize(img, img, cv::Size(224, 224));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    std::vector<float> input_vals(1 * 3 * 224 * 224);
    int idx = 0;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                float v = img.at<cv::Vec3f>(y, x)[c];
                float mean = (c==0?0.485f:(c==1?0.456f:0.406f));
                float std  = (c==0?0.229f:(c==1?0.224f:0.225f));
                input_vals[idx++] = (v - mean) / std;
            }
        }
    }

    std::vector<int64_t> shape = {1,3,224,224};
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input_vals.data(),
        input_vals.size(),
        shape.data(),
        shape.size()
    );

    const char* in_names[]  = { input_name.c_str() };
    const char* out_names[] = { output_name.c_str() };

    auto outputs = session_ptr->Run(
        Ort::RunOptions{nullptr},
        in_names,
        &input_tensor,
        1,
        out_names,
        1
    );

    float* logits = outputs[0].GetTensorMutableData<float>();
    auto probs = softmax(logits, 10);

    int best = 0;
    for (int i = 1; i < 10; i++)
        if (probs[i] > probs[best]) best = i;

    confidence = probs[best];
    return best;
}

// ---------------- Main ----------------
int main() {
    session_options.SetIntraOpNumThreads(1);

    session_ptr = new Ort::Session(
        env,
        "../models/mobilenetv2_cifar10.onnx",
        session_options
    );

    input_name  = session_ptr->GetInputNameAllocated(0, allocator).get();
    output_name = session_ptr->GetOutputNameAllocated(0, allocator).get();

    httplib::Server svr;

    svr.Post("/infer", [](const httplib::Request& req, httplib::Response& res) {
        if (req.body.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"empty body\"}", "application/json");
            return;
        }

        std::vector<uchar> buf(req.body.begin(), req.body.end());
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);

        float confidence = 0.0f;
        auto start = std::chrono::high_resolution_clock::now();

        int cls = run_inference_from_mat(img, confidence);

        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        if (cls < 0) {
            res.status = 400;
            res.set_content("{\"error\":\"bad image\"}", "application/json");
            return;
        }

        res.set_content(
            std::string("{\"label\":\"") + CIFAR10_LABELS[cls] +
            "\", \"confidence\": " + std::to_string(confidence) +
            ", \"latency_ms\": " + std::to_string(latency_ms) + "}",
            "application/json"
        );
    });


    std::cout << "Server running on http://localhost:8080\n";
    svr.listen("0.0.0.0", 8080);
}
