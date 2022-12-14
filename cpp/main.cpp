#include <cstdint>
#include <cmath>
#include <algorithm>
#include "Utils.hpp"
#include <opencv2/opencv.hpp>

static cv::Mat currentCubemap;

inline std::pair<float, float> mapsToCube(float x, float y, float z) {

    float offset = 1;

    if (std::fabs(x) < std::fabs(y)) {
        std::swap(x, y);
        offset = 3;
    }

    if (std::fabs(x) < std::fabs(z)) {
        std::swap(x, z);
        if (offset < 2) std::swap(y, z);
        offset = 5;
    }

    float const norm = 1.0f / std::fabs(x);

    x *= norm;
    y *= norm;
    z *= norm;
    
    return {
        (1 + y) / 2,
        offset + (z - x) / 2,
    };
}


template<typename T>
struct Vec3 {
    T x, y, z;
};

struct Mat3 {
    float m[9];

    Mat3 operator*(Mat3 const& other) const {
        Mat3 res;

        res.m[0] = m[0] * other.m[0] + m[1] * other.m[3] + m[2] * other.m[6];
        res.m[1] = m[0] * other.m[1] + m[1] * other.m[4] + m[2] * other.m[7];
        res.m[2] = m[0] * other.m[2] + m[1] * other.m[5] + m[2] * other.m[8];

        res.m[3] = m[3] * other.m[0] + m[4] * other.m[3] + m[5] * other.m[6];
        res.m[4] = m[3] * other.m[1] + m[4] * other.m[4] + m[5] * other.m[7];
        res.m[5] = m[3] * other.m[2] + m[4] * other.m[5] + m[5] * other.m[8];

        res.m[6] = m[6] * other.m[0] + m[7] * other.m[3] + m[8] * other.m[6];
        res.m[7] = m[6] * other.m[1] + m[7] * other.m[4] + m[8] * other.m[7];
        res.m[8] = m[6] * other.m[2] + m[7] * other.m[5] + m[8] * other.m[8];

        return res;
    }

    template<typename T>
    Vec3<float> operator*(Vec3<T> const& v) const {
        return {
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[3] * v.x + m[4] * v.y + m[5] * v.z,
            m[6] * v.x + m[7] * v.y + m[8] * v.z,
        };
    }
};



extern "C" void viewerQuery(
    std::uint8_t* dst, 
    int dstWidth, int dstHeight, 
    float theta0, float phi0, float gamma, float f
) {

    const int nCubeSide     = currentCubemap.cols;

    using std::cos;
    using std::sin;


    const Mat3 RotZ{
        cos(theta0), -sin(theta0), 0,
        sin(theta0), cos(theta0) , 0,
        0          , 0           , 1
    };

    const Mat3 RotY{
        cos(phi0) , 0, sin(phi0),
        0         , 1, 0,
        -sin(phi0), 0, cos(phi0)
    };

    const Mat3 Tilt{
        cos(gamma), -sin(gamma), 0,
        sin(gamma), cos(gamma) , 0,
        0         , 0          , 1
    };

    const Mat3 projection{
        2 * f / dstWidth, 0               , -f * dstHeight / dstWidth,
        0               , 2 * f / dstWidth, -f,
        0               , 0               , 1
    };

    const Mat3 transform = RotZ * RotY * Tilt * projection;


    for (int i = 0; i < dstHeight; ++i) {
        for (int j = 0; j < dstWidth; ++j) {

            auto const [Rotx, Roty, Rotz] = transform * Vec3<int>{i, j, 1};


            auto [srci, srcj] = mapsToCube(Rotx, Roty, Rotz);

            const int ii = std::clamp(srci * nCubeSide, 0.0f, nCubeSide - 1.0f);
            const int jj = std::clamp(srcj * nCubeSide, 0.0f, 6 * nCubeSide - 1.0f);

            std::uint8_t*       dstPixelRGBA = &dst[4 * (i * dstWidth + j)];
            std::uint8_t const* srcPixelRGB = &currentCubemap.data[3 * (nCubeSide * jj + ii)];

            std::copy_n(srcPixelRGB, 3, dstPixelRGBA);
            dstPixelRGBA[3] = 255;

        }
    }

}

struct jsOut {
    std::uint32_t size;
    uchar* data;
};
#include <iostream>

extern "C" jsOut const* jsCubeMap(std::uint8_t* srcBuffer, int size, int downScale) {


    cv::Mat temp(1, size, CV_8U, srcBuffer);

    cv::Mat srcMat = cv::imdecode(temp, cv::IMREAD_COLOR);

    
    const int nCubeSide = srcMat.rows / downScale;

    std::uint8_t* dstBuffer = (std::uint8_t*)malloc(3 * nCubeSide * 6 * nCubeSide);
    
    RGBA src(srcMat.data, srcMat.cols, srcMat.rows);
    RGBA dst(dstBuffer, nCubeSide, 6 * nCubeSide);

    toCubeMapFace<LEFT>(src, dst, nCubeSide);
    toCubeMapFace<FRONT>(src, dst, nCubeSide);
    toCubeMapFace<RIGHT>(src, dst, nCubeSide);
    toCubeMapFace<BACK>(src, dst, nCubeSide);
    toCubeMapFace<TOP>(src, dst, nCubeSide);
    toCubeMapFace<BOTTOM>(src, dst, nCubeSide);

    currentCubemap = cv::Mat(6 * nCubeSide, nCubeSide, CV_8UC3);
    std::copy_n(dstBuffer, 3 * nCubeSide * 6 * nCubeSide, currentCubemap.data);
    free(dstBuffer);

    static std::vector<uchar> outBuffer;
    static jsOut out;
    uint32_t dummy;

    outBuffer.clear();
    std::vector<uchar> left, front, right, back, top, bottom;
    cv::imencode(".jpg", currentCubemap(cv::Range(0 * nCubeSide, 1 * nCubeSide), cv::Range(0, nCubeSide)), left);
    cv::imencode(".jpg", currentCubemap(cv::Range(1 * nCubeSide, 2 * nCubeSide), cv::Range(0, nCubeSide)), front);
    cv::imencode(".jpg", currentCubemap(cv::Range(2 * nCubeSide, 3 * nCubeSide), cv::Range(0, nCubeSide)), right);
    cv::imencode(".jpg", currentCubemap(cv::Range(3 * nCubeSide, 4 * nCubeSide), cv::Range(0, nCubeSide)), back);
    cv::imencode(".jpg", currentCubemap(cv::Range(4 * nCubeSide, 5 * nCubeSide), cv::Range(0, nCubeSide)), top);
    cv::imencode(".jpg", currentCubemap(cv::Range(5 * nCubeSide, 6 * nCubeSide), cv::Range(0, nCubeSide)), bottom);

    outBuffer.resize(outBuffer.size() + 4);
    dummy = left.size();
    std::memcpy(&outBuffer.back() - 3, &dummy, 4);
    outBuffer.insert(outBuffer.end(), left.begin(), left.end());

    outBuffer.resize(outBuffer.size() + 4);
    dummy = front.size();
    std::memcpy(&outBuffer.back() - 3, &dummy, 4);
    outBuffer.insert(outBuffer.end(), front.begin(), front.end());

    outBuffer.resize(outBuffer.size() + 4);
    dummy = right.size();
    std::memcpy(&outBuffer.back() - 3, &dummy, 4);
    outBuffer.insert(outBuffer.end(), right.begin(), right.end());

    outBuffer.resize(outBuffer.size() + 4);
    dummy = back.size();
    std::memcpy(&outBuffer.back() - 3, &dummy, 4);
    outBuffer.insert(outBuffer.end(), back.begin(), back.end());

    outBuffer.resize(outBuffer.size() + 4);
    dummy = top.size();
    std::memcpy(&outBuffer.back() - 3, &dummy, 4);
    outBuffer.insert(outBuffer.end(), top.begin(), top.end());

    outBuffer.resize(outBuffer.size() + 4);
    dummy = bottom.size();
    std::memcpy(&outBuffer.back() - 3, &dummy, 4);
    outBuffer.insert(outBuffer.end(), bottom.begin(), bottom.end());

    out.size = outBuffer.size();
    out.data = outBuffer.data();

    cv::cvtColor(currentCubemap, currentCubemap, cv::COLOR_BGR2RGB);
    
    return &out;
}