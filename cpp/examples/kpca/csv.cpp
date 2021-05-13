
#include <fstream>
#include <iostream>
#include <sstream>   // std::stringstream
#include <stdexcept> // std::runtime_error
#include <string>
#include <utility> // std::pair
#include <vector>
class csvInfo {
public:
  std::vector<float> matrix;
  int cols;
  int rows;
  std::vector<std::string> column_names;
};
/**
 * Based on this:
 * https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/
 */
csvInfo read_csv(std::string filename, bool ignore_last_col = true,
                 bool col_major = true) {
  // Reads a CSV file into a vector of <string, vector<int>> pairs where
  // each pair represents <column name, column values>

  // Create a vector of <string, int vector> pairs to store the result
  std::vector<std::string> col_names;
  csvInfo csv;
  // Create an input filestream
  std::ifstream myFile(filename);
  int num_lines = 0;
  int num_cols = 0;
  // Make sure the file is open
  if (!myFile.is_open())
    throw std::runtime_error("Could not open file");

  // Helper vars
  std::string line, colname;
  float val;

  // Read the column names
  if (myFile.good()) {
    // Extract the first line in the file
    std::getline(myFile, line);

    // Create a stringstream from line
    std::stringstream ss(line);

    // Extract each column name
    while (std::getline(ss, colname, ',')) {

      // Initialize and add <colname, int vector> pairs to result
      col_names.push_back(colname);
    }
  }
  while (std::getline(myFile, line))
    ++num_lines;
  if (ignore_last_col) {
    col_names.pop_back();
  }
  num_cols = col_names.size();
  myFile.clear();
  myFile.seekg(0);
  std::getline(myFile, line);
  // Read data, line by line
  int index = 0;
  int curr_col = 0;
  while (std::getline(myFile, line)) {
    // Create a stringstream of the current line
    std::stringstream ss(line);

    // Extract each float
    while (ss >> val) {
      if (curr_col < num_cols) {
        csv.matrix.push_back(val);
      }

      // If the next token is a comma, ignore it and move on
      if (ss.peek() == ',')
        ss.ignore();

      // Increment the column index
      index++;
      curr_col++;
      if (curr_col == num_cols) {
        std::string temp;
        ss >> temp;
      }
    }
    curr_col = 0;
  }

  // Close file
  myFile.close();
  // csv.matrix = matrix;
  csv.cols = num_cols;
  csv.rows = num_lines;
  csv.column_names = col_names;
  return csv;
}

void write_matrix_csv(std::string filename, float *matrix, int rows, int cols,
                      bool write_col_names = true) {
  std::ofstream myFile(filename);

  // Send column names to the stream
  if (write_col_names) {
    for (int j = 0; j < cols; j++) {
      myFile << j;
      if (j != cols - 1)
        myFile << ","; // No comma at end of line
    }
    myFile << "\n";
  }

  // Send data to the stream
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; ++j) {
      myFile << matrix[i * cols + j];
      if (j != cols - 1)
        myFile << ","; // No comma at end of line
    }
    myFile << "\n";
    myFile.flush();// TODO try flushing
  }
  // Close the file
  myFile.close();
}
