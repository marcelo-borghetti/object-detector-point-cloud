#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

class LogsFile
{
private:
	stringstream baseDir;
	ofstream myfile;
	stringstream filename;

public:
	void createStructure(string numDir);
	void createDir(string nameDir);
	void add(string str);
};
