#ifndef GEMS_PYKVRIGIDREGISTRATION_H
#define GEMS_PYKVLRIGIDREGISTRATION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "itkImage.h"
#include "itkVersorRigid3DTransform.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "kvlRegisterImages.h"

class KvlRigidRegistration {
    typedef itk::Image< double, 3 > ImageType;
    typedef itk::VersorRigid3DTransform< double > RigidTransformType;
    typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType, ImageType> MetricType;
    kvl::RegisterImages<RigidTransformType, MetricType>::Pointer registerer;
    public:
        // Python accessible
        KvlRigidRegistration()
        {
            registerer = kvl::RegisterImages<RigidTransformType, MetricType>::New();
            //Set MI histogram bins
            registerer->GetMetric()->SetNumberOfHistogramBins(64);
            registerer->SetCenterOfMassInit(false);
        }
    
        KvlRigidRegistration(double transScale, int numIter,
        int numHistBins, py::array_t<double> shrinkScales, double bgLevel,
        py::array_t<double> smoothSigmas, bool useCenterOfMass, double sampRate,
        std::string interpolator)
        {
        
            py::buffer_info scales_info  = shrinkScales.request();
            const int numScales = scales_info.shape[0];
            std::vector<double> scales;
            
            for ( int scaleNum = 0; scaleNum < numScales; scaleNum++ ) {
            	 scales.push_back( shrinkScales.at(scaleNum) );
            }

            py::buffer_info sigmas_info  = smoothSigmas.request();
            const int numSigmas = sigmas_info.shape[0];
            std::vector<double> sigmas;
            
            for ( int sigmaNum = 0; sigmaNum < numSigmas; sigmaNum++ ) {
            	 sigmas.push_back( smoothSigmas.at(sigmaNum) );
            }
            
            registerer = kvl::RegisterImages<RigidTransformType, MetricType>::New();
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

        py::array_t<double> GetFinalTransformation();
};
#endif //GEMS_PYKVLRIGIDREGISTRATION_H
