#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

struct ImageQualityMetrics {
    double mse = 0.0;
    double psnr = 100.0;
    double ssim = 1.0;
};

class ImageMetrics {
public:
    static ImageQualityMetrics compute(
        const std::vector<float>& imgA,
        const std::vector<float>& imgB,
        uint32_t width,
        uint32_t height
    ) {
        ImageQualityMetrics result{};
        size_t pixelCount = static_cast<size_t>(width) * height;
        if (pixelCount == 0 || imgA.size() < pixelCount || imgB.size() < pixelCount) {
            return result;
        }

        double sumSqErr = 0.0;
        for (size_t i = 0; i < pixelCount; i++) {
            double diff = static_cast<double>(imgA[i]) - static_cast<double>(imgB[i]);
            sumSqErr += diff * diff;
        }

        result.mse = sumSqErr / static_cast<double>(pixelCount);
        if (result.mse < 1e-12) {
            result.psnr = 100.0;
        } else {
            result.psnr = 10.0 * std::log10(1.0 / result.mse);
        }

        // SSIM computation using 8x8 blocks
        const int blockSize = 8;
        const double c1 = (0.01 * 1.0) * (0.01 * 1.0);
        const double c2 = (0.03 * 1.0) * (0.03 * 1.0);

        int numBlocksX = width / blockSize;
        int numBlocksY = height / blockSize;
        if (numBlocksX == 0 || numBlocksY == 0) {
            result.ssim = 1.0;
            return result;
        }

        double totalSsim = 0.0;
        int validBlocks = 0;

        for (int by = 0; by < numBlocksY; by++) {
            for (int bx = 0; bx < numBlocksX; bx++) {
                double meanA = 0.0;
                double meanB = 0.0;
                int startX = bx * blockSize;
                int startY = by * blockSize;

                for (int y = 0; y < blockSize; y++) {
                    for (int x = 0; x < blockSize; x++) {
                        int idx = (startY + y) * width + (startX + x);
                        meanA += imgA[idx];
                        meanB += imgB[idx];
                    }
                }
                meanA /= (blockSize * blockSize);
                meanB /= (blockSize * blockSize);

                double varA = 0.0;
                double varB = 0.0;
                double covAB = 0.0;

                for (int y = 0; y < blockSize; y++) {
                    for (int x = 0; x < blockSize; x++) {
                        int idx = (startY + y) * width + (startX + x);
                        double diffA = imgA[idx] - meanA;
                        double diffB = imgB[idx] - meanB;
                        varA += diffA * diffA;
                        varB += diffB * diffB;
                        covAB += diffA * diffB;
                    }
                }
                varA /= (blockSize * blockSize - 1);
                varB /= (blockSize * blockSize - 1);
                covAB /= (blockSize * blockSize - 1);

                double ssimNum = (2.0 * meanA * meanB + c1) * (2.0 * covAB + c2);
                double ssimDenom = (meanA * meanA + meanB * meanB + c1) * (varA + varB + c2);
                double blockSsim = ssimNum / ssimDenom;

                totalSsim += std::clamp(blockSsim, 0.0, 1.0);
                validBlocks++;
            }
        }

        if (validBlocks > 0) {
            result.ssim = totalSsim / validBlocks;
        }

        return result;
    }
};
