#include "classifier.h"

/*#include <algorithm>
#include <iosfwd>
#include <memory>

#include <utility>

#include <fstream>
#include <map>

*/

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <xopenme.h>

using namespace std;
namespace fs = boost::filesystem;

int getenv_i(const char* name, int def) {
  return getenv(name) ? atoi(getenv(name)) : def;
}

string getenv_s(const char* name) {
  const char* val = getenv(name);
  assert(val && strlen(val) > 0);
  return string(val);
}

const int BATCH_COUNT = getenv_i("CK_BATCH_COUNT", 1);
const int BATCH_SIZE = getenv_i("CK_CAFFE_BATCH_SIZE", 1);
const int IMAGES_COUNT = BATCH_COUNT * BATCH_SIZE;
const int SKIP_IMAGES = getenv_i("CK_SKIP_IMAGES", 0);
const string IMAGES_DIR = getenv_s("CK_ENV_DATASET_IMAGENET_VAL");

vector<string> get_images() {
  const string filter1(".JPG");
  const string filter2(".JPEG");

  vector<string> all_images;
  fs::directory_iterator end_iter;
  for (fs::directory_iterator dir_iter(IMAGES_DIR) ; dir_iter != end_iter ; ++dir_iter){
    if (!fs::is_regular_file(dir_iter->status())) continue;

    string file = dir_iter->path().string();

    string tmp = boost::to_upper_copy(file);
    if (tmp.compare(file.size()-4, 4, filter1) != 0 &&
        tmp.compare(file.size()-5, 5, filter2) != 0) continue;

    all_images.push_back(file);
  }

  sort(all_images.begin(), all_images.end());

  int last_index = SKIP_IMAGES + IMAGES_COUNT;
  if (last_index >= all_images.size())
    last_index = all_images.size()-1;

  return vector<string>(&all_images[SKIP_IMAGES], &all_images[last_index]);
}

enum {
  TIMER_INIT_CLASSIFIER,
  TIMER_LOAD_IMAGE,
  TIMER_CLASSIFY_IMAGE,

  TIMERS_COUNT
};

int main(int argc, char** argv) {
  xopenme_init(TIMERS_COUNT, 0);

  ::google::InitGoogleLogging(argv[0]);

  string model_file = getenv_s("CK_CAFFE_MODEL_FILE");
  string weights_file = getenv_s("CK_ENV_MODEL_CAFFE_WEIGHTS");

  fs::path aux_path(getenv_s("CK_ENV_DATASET_IMAGENET_AUX"));
  string mean_file = (aux_path / "imagenet_mean.binaryproto").native();
  string labels_file = (aux_path / "synset_words.txt").native();

  cout << "Model file: " << model_file << endl;
  cout << "Weights file: " << weights_file << endl;
  cout << "Mean file: " << mean_file << endl;
  cout << "Labels file: " << labels_file << endl;
  cout << "Images dir: " << IMAGES_DIR << endl;
  cout << "Batch count: " << BATCH_COUNT << endl;
  cout << "Batch size: " << BATCH_SIZE << endl;

  // Load processing image filenames
  vector<string> images = get_images();

  // Build net
  xopenme_clock_start(TIMER_INIT_CLASSIFIER);
  Classifier classifier(model_file, weights_file, mean_file, labels_file);
  xopenme_clock_end(TIMER_INIT_CLASSIFIER);
  cout << "Classifier initialised in " << xopenme_get_timer(TIMER_INIT_CLASSIFIER) << "s" << endl;

  // Run batched mode
  double load_total_time = 0;
  double class_total_time = 0;
  int image_index = 0;
  int images_processed = 0;
  for (int batch_index = 0; batch_index < BATCH_COUNT; batch_index++) {
    cout << "Batch " << batch_index << endl;

    // Classify batch
    for (int i = 0; i < BATCH_SIZE; i++) {
      // Load image
      xopenme_clock_start(TIMER_LOAD_IMAGE);
      cv::Mat img = cv::imread(images[image_index], -1);
      cv::Mat img_prepared = classifier.PrepareImage(img);
      xopenme_clock_end(TIMER_LOAD_IMAGE);   
      load_total_time += xopenme_get_timer(TIMER_LOAD_IMAGE);
      CHECK(!img.empty()) << "Unable to decode image " << images[image_index];

      // Classify
      xopenme_clock_start(TIMER_CLASSIFY_IMAGE);
      vector<float> probs = classifier.Predict(img_prepared);
      xopenme_clock_end(TIMER_CLASSIFY_IMAGE);

      // Print the top N predictions.
      vector<Prediction> predictions = classifier.ProcessPredictions(probs);
      for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
        std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
      }

      // Exclude first batch from averaging
      if (batch_index > 0 || BATCH_COUNT == 1) {
        class_total_time += xopenme_get_timer(TIMER_CLASSIFY_IMAGE);
        images_processed += BATCH_SIZE;
      }

      image_index++;
    }
  }

  double class_avg_time = class_total_time / double(images_processed);

  cout << endl;
  cout << "Images processed: " << images_processed << endl;
  cout << "All images loaded in " << load_total_time << "s" << endl;
  cout << "All images classified in " << class_total_time << "s" << endl;
  cout << "Average classification time: " << class_avg_time << "s";
  if (BATCH_COUNT > 1) cout << " (first batch excluded)";
  cout << endl;

  xopenme_dump_state();
  xopenme_finish();
  return 0;
}
