#ifndef GEMS_PYKVLAFFINEREGISTRATION_H
#define GEMS_PYKVLAFFINEREGISTRATION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "itkImage.h"
#include "kvlRegisterImages.h"
#include "itkAffineTransform.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"

class KvlAffineRegistration {
    typedef itk::Image< double, 3 > ImageType;
    typedef itk::AffineTransform< double, 3 >  AffineTransformType;
    typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType, ImageType> MetricType;
    kvl::RegisterImages<AffineTransformType, MetricType>::Pointer registerer;
    public:
        // Python accessible
        KvlAffineRegistration()
        {
            registerer = kvl::RegisterImages<AffineTransformType, MetricType>::New();
            // Set MI histogram bins
            registerer->GetMetric()->SetNumberOfHistogramBins(64);
            registerer->SetCenterOfMassInit(false);
        }
    
        KvlAffineRegistration(double transScale, int numIter,
        int numHistBins, py::array_t<double> shrinkScales, double bgLevel,
        py::array_t<double> smoothSigmas, bool useCenterOfMass, double sampRate,
        std::string interpolator)
        {
        
            py::buffer_info scales_info  = shrinkScales.request();
            const int numScales = scales_info.shape[0];
            std::vector<double> scales;
            
            for ( int scaleNum = 0; scaleNum < numScales; ++scaleNum ) {
            	 scales.push_back( scales.at(scaleNum) );
            }

            py::buffer_info sigmas_info  = smoothSigmas.request();
            const int numSigmas = sigmas_info.shape[0];
            std::vector<double> sigmas;
            
            for ( int sigmaNum = 0; sigmaNum < numSigmas; sigmaNum++ ) {
            	 sigmas.push_back( smoothSigmas.at(sigmaNum) );
            }
            
            registerer = kvl::RegisterImages<AffineTransformType, MetricType>::New();
            registerer->SetTranslationScale(transScale);
            registerer->SetNumberOfIterations(numIter);
            registerer->SetShrinkScales(scales);
            registerer->SetBackgroundGrayLevel(bgLevel);
            registerer->SetSmoothingSigmas(sigmas);
            registerer->SetCenterOfMassInit(useCenterOfMass);
            registerer->SetSamplingPercentage(sampRate);
            registerer->SetInterpolator(interpolator);
            registerer->GetMetric()->SetNumberOfHistogramBins(numHistBins);
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

        py::array_t<double> GetFinalTransformationMatrix();
};
#endif //GEMS_PYKVLAFFINEREGISTRATION_H
