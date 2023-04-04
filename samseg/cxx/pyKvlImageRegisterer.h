#ifndef GEMS_PYKVLIMAGEREGISTERER_H
#define GEMS_PYKVLIMAGEREGISTERER_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "kvlRegisterImages.h"

class KvlImageRegisterer {
    kvl::RegisterImages::Pointer registerer;
    public:
        // Python accessible
        KvlImageRegisterer()
        {
            registerer = kvl::RegisterImages::New();
        }
    
        KvlImageRegisterer(double transScale, int numIter,
        int numHistBins, int numLevels, double bgLevel,
        double sigma, bool useCenterOfMass, double sampRate,
        std::string interpolator)
        {
            registerer = kvl::RegisterImages::New();
            registerer->SetTranslationScale(transScale);
            registerer->SetNumberOfIterations(numIter);
            registerer->SetNumberOfHistogramBins(numHistBins);
            registerer->SetNumberOfLevels(numLevels);
            registerer->SetBackgroundGrayLevel(bgLevel);
            registerer->SetSmoothingSigmas(sigma);
            registerer->SetCenterOfMassInit(useCenterOfMass);
            registerer->SetSamplingPercentage(sampRate);
            registerer->SetInterpolator(interpolator);
        }

        void ReadImages(const std::string &fileNameT1, const std::string &fileNameT2)
        {
            registerer->ReadImages(fileNameT1.c_str(), fileNameT2.c_str());
        }

        void InitializeTransform()
        {
            registerer->InitializeTransform();
        }

        void Register()
        {
            registerer->RunRegistration();
        }

        void WriteOutResults(std::string outImageFile)
        {
            registerer->WriteOutResults(outImageFile);
        }

};
#endif //GEMS_PYKVLIMAGEREGISTERER_H
