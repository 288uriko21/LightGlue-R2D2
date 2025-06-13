#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// Сохранение ключевых точек и дескрипторов в файл
void savePointsAndDescriptors(const vector<KeyPoint>& keypoints, const Mat& descriptors, const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Не удалось открыть файл для записи." << endl;
        return;
    }

    // Сохраняем количество точек
    int num_keypoints = keypoints.size();
    file.write(reinterpret_cast<const char*>(&num_keypoints), sizeof(int));

    // Сохраняем координаты точек
    for (const auto& kp : keypoints) {
        file.write(reinterpret_cast<const char*>(&kp.pt.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&kp.pt.y), sizeof(float));
    }

    // Сохраняем дескрипторы
    file.write(reinterpret_cast<const char*>(descriptors.data), descriptors.total() * descriptors.elemSize());

    file.close();
    cout << "Ключевые точки и дескрипторы сохранены в " << filename << endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Использование: ./surf_extractor <входное_изображение> <выходной_файл>" << endl;
        return -1;
    }

    string imagePath = argv[1];
    string outputFile = argv[2];

    // Загружаем изображение
    Mat img = imread(imagePath, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Ошибка загрузки изображения." << endl;
        return -1;
    }

    // Инициализируем SURF
    Ptr<SURF> surf = SURF::create(600);

    vector<KeyPoint> keypoints;
    Mat descriptors;
    surf->detectAndCompute(img, noArray(), keypoints, descriptors);

    cout << "Найдено " << keypoints.size() << " ключевых точек." << endl;

    savePointsAndDescriptors(keypoints, descriptors, outputFile);

    return 0;
}
