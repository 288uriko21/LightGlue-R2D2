//g++ -o check_surf Csurf.cpp $(pkg-config --cflags --libs opencv4) -lopencv_xfeatures2d


#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main() {
    // Загружаем изображение
    Mat img = imread("COCO_train2014_000000005811.jpg", IMREAD_GRAYSCALE);
    cout << -1;
    if (img.empty()) {
        cout << "Ошибка загрузки изображения!" << endl;
        return -1;
    }
    cout << 0;
    // Создаём объект SURF
    Ptr<SURF> detector = SURF::create(600);

    // Находим ключевые точки и дескрипторы
    cout << 1;
    vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);

    cout << "Найдено ключевых точек: " << keypoints.size() << endl;

    // Рисуем ключевые точки
    Mat img_keypoints;
    drawKeypoints(img, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    // Сохраняем результат
    imwrite("output.jpg", img_keypoints);
    cout << "Результат сохранён: output.jpg" << endl;

    return 0;
}
// #include <opencv2/opencv.hpp>
// #include <opencv2/xfeatures2d.hpp>

// int main() {
//     if(cv::xfeatures2d::SURF::create() != nullptr) {
//         std::cout << "SURF доступен!" << std::endl;
//     } else {
//         std::cout << "SURF не доступен!" << std::endl;
//     }
//     return 0;
// }

