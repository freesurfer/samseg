#ifndef kvlRegisterImages_hxx
#define kvlRegisterImages_hxx

#include "kvlRegisterImages.h"
#include "itkImageFileReader.h"
#include "itkCenteredTransformInitializer.h"
#include "itkResampleImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkNormalizeImageFilter.h"


namespace kvl
{

//
//
//
template< typename TransformationType, typename MetricType >
RegisterImages<TransformationType, MetricType>
::RegisterImages()
{
    m_FixedImage = 0;
    m_MovingImage = 0;
    m_InitialTransform = TransformationType::New();
    m_InitialTransform->SetIdentity();
    m_FinalTransform = 0;
    //This can be changed using the setter later on
    m_CenterOfMass = false;
    m_Metric = MetricType::New();
    m_TranslationScale = -100;
    m_NumberOfIterations = 100;
    m_BackgroundGrayLevel = 0;
    m_SamplingPercentage = 0.5;
    m_Interpolator = "b";
    m_ShrinkScales.push_back(2.0);
    m_ShrinkScales.push_back(1.0);
    m_ShrinkScales.push_back(0.0);
    m_SmoothingSigmas.push_back(4.0);
    m_SmoothingSigmas.push_back(2.0);
    m_SmoothingSigmas.push_back(0.0);
    
}

template< typename TransformationType, typename MetricType >
void RegisterImages<TransformationType, MetricType>
::ReadImages(const char* fileNameT1, const char* fileNameT2)
{
    //Read in the T1 image
    typedef itk::ImageFileReader< ImageType >  ReaderType;
    ReaderType::Pointer  fixedImageReader = ReaderType::New();
    fixedImageReader->SetFileName( fileNameT1 );
    fixedImageReader->Update(); 
    //Set-up the fixed image
    m_FixedImage = fixedImageReader->GetOutput();

    //Then the T2
    ReaderType::Pointer  movingImageReader = ReaderType::New();
    movingImageReader->SetFileName( fileNameT2 );
    movingImageReader->Update();
    //Set-up the moving image
    m_MovingImage = movingImageReader->GetOutput();
}



template< typename TransformationType, typename MetricType >
void RegisterImages<TransformationType, MetricType>
::InitializeTransform()
{
   typedef itk::CenteredTransformInitializer<TransformationType, ImageType, ImageType> TransformInitializerType;
   typename TransformInitializerType::Pointer initializer = TransformInitializerType::New();
   initializer->SetTransform(m_InitialTransform);
   initializer->SetFixedImage(m_FixedImage);
   initializer->SetMovingImage(m_MovingImage);

   if(m_CenterOfMass)
   {
        initializer->MomentsOn();
   }
   else
   {
        initializer->GeometryOn();
   }
   initializer->InitializeTransform();

   typename TransformationType::MatrixType initialMatrix = m_InitialTransform->GetMatrix();
   typename TransformationType::OffsetType initialOffset = m_InitialTransform->GetOffset();
   std::cout << "Initial matrix = " << std::endl
             << initialMatrix << std::endl;
   std::cout << "Initial offset = " << std::endl
             << initialOffset << "\n"
             << std::endl;

}


template <typename TransformationType, typename MetricType>
void RegisterImages<TransformationType, MetricType>
::RunRegistration()
{
    //Set-up registration and optimization
    typedef itk::ImageRegistrationMethodv4<ImageType, ImageType, TransformationType> RegistrationType;

    typename RegistrationType::Pointer registration = RegistrationType::New();
    OptimizerType::Pointer optimizer = OptimizerType::New();
    registration->SetOptimizer(optimizer);

    registration->SetMetric(m_Metric);

    //Normalize images
    typedef itk::NormalizeImageFilter<ImageType,ImageType> NormalizeFilterType;
    NormalizeFilterType::Pointer fixedNormalizer = NormalizeFilterType::New();
    NormalizeFilterType::Pointer movingNormalizer = NormalizeFilterType::New();

    fixedNormalizer->SetInput(m_FixedImage);
    movingNormalizer->SetInput(m_MovingImage);
    fixedNormalizer->Update();
    movingNormalizer->Update();

    //Feed the normalized images to the registration
    registration->SetFixedImage(fixedNormalizer->GetOutput());
    registration->SetMovingImage(movingNormalizer->GetOutput());

    //Set the initial transform
    registration->SetInitialTransform(m_InitialTransform);

    std::cout << "\nInitial transform: " << std::endl;
    std::cout << "   Moving image: " << registration->GetInitialTransform()->GetParameters() << std::endl;
    std::cout << "   Fixed image:  " << registration->GetInitialTransform()->GetFixedParameters() << "\n"
                << std::endl;

    //Get the optimizer scales and set them
    typedef itk::RegistrationParameterScalesFromPhysicalShift<MetricType> ScalesEstimatorType;
    auto scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric(m_Metric);
    scalesEstimator->SetTransformForward(true);
    
    OptimizerScalesType optimizerScales(m_InitialTransform->GetNumberOfParameters());
    optimizerScales.fill(1.0);
    //Fill in the last three elements with the translation scales
    optimizerScales[optimizerScales.Size()-1] = m_TranslationScale;
    optimizerScales[optimizerScales.Size()-2] = m_TranslationScale;
    optimizerScales[optimizerScales.Size()-3] = m_TranslationScale;

    // Set the other parameters of optimizer
    optimizer->SetDoEstimateLearningRateOnce(false);
    optimizer->SetDoEstimateLearningRateAtEachIteration(true);
    // Software Guide : EndCodeSnippet
 
    // Set the other parameters of optimizer
    optimizer->SetLowerLimit(0);
    optimizer->SetUpperLimit(2);
    optimizer->SetEpsilon(0.2);
    optimizer->SetNumberOfIterations(m_NumberOfIterations);
    optimizer->SetMinimumConvergenceValue(1e-4);
    optimizer->SetConvergenceWindowSize(5);
 

    if (m_TranslationScale > 0){
       std::cout << "Using optimizer scales:" << std::endl;
       std::cout << optimizerScales << std::endl;
       optimizer->SetScales(optimizerScales);
    }
    else{
       std::cout << "Using scales estimator." << std::endl;
       optimizer->SetScalesEstimator(scalesEstimator);
    }

    //Could be skipped if we don't need to test
    constexpr int randomNumberGeneratorSeed = 121213;
    registration->MetricSamplingReinitializeSeed(randomNumberGeneratorSeed);

    //Set sampling strategy
    typename RegistrationType::MetricSamplingStrategyType samplingStrategy = RegistrationType::RANDOM;
    registration->SetMetricSamplingStrategy(samplingStrategy);
    registration->SetMetricSamplingPercentage(m_SamplingPercentage);


    //Add the observer
    CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);

    //Add number of levels
    registration->SetNumberOfLevels(m_ShrinkScales.size());
    typename RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize(m_ShrinkScales.size());

    for (auto i = 0; i < m_ShrinkScales.size(); ++i)
    {
        shrinkFactorsPerLevel[i] = static_cast<unsigned long>(pow(2, m_ShrinkScales.at(i)));
    }
    registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

    // Set the smoothing sigmas for each level in terms of voxels.
    registration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(true);
    typename RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize(m_ShrinkScales.size());
    for (auto i = 0; i < m_ShrinkScales.size(); ++i)
    {
        smoothingSigmasPerLevel[i] = m_SmoothingSigmas.at(i);
    }

    registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);

    std::cout<< "Metric:" << std::endl;
    m_Metric->Print(std::cout, 0);
    // Create the Command observer and register it with the registration.
    typedef RegistrationInterfaceCommand<RegistrationType> CommandType;
    typename CommandType::Pointer command = CommandType::New();
    registration->AddObserver(itk::MultiResolutionIterationEvent(), command);

    //  Trigger the registration process by calling \code{Update()}.
    try
    {
        registration->Update();

        std::string stopCondition{registration->GetOptimizer()->GetStopConditionDescription()};
        std::cout << std::endl;

        if (((stopCondition.find("Step too small") != std::string::npos) || (stopCondition.find("Convergence checker passed at iteration") != std::string::npos)) == false)
        {
            std::cout << "Warning message:" << std::endl;
            std::cout << "   Model failed to converge!" << std::endl;

            if (stopCondition.find("Maximum number of iterations") != std::string::npos)
            {
                std::cout << "   Try to increase the number of iterations by adding the option \"-i 500\" by the end of the command" << std::endl;
            }
        }

        std::cout << "   Optimizer stop condition: " << stopCondition << std::endl;
    }
    catch (itk::ExceptionObject &err)
    {
        std::cout << "ExceptionObject caught !" << std::endl;
        std::cout << err << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cout << "Caught an non-ITK exception !" << std::endl;
        exit(EXIT_FAILURE);
    }

    const typename TransformationType::ParametersType finalParameters =
      registration->GetOutput()->Get()->GetParameters();

    const unsigned int currentIterations = optimizer->GetCurrentIteration();
    const double bestValue = optimizer->GetValue();

    m_FinalTransform = TransformationType::New();
    m_FinalTransform->SetFixedParameters(
      registration->GetOutput()->Get()->GetFixedParameters());
    m_FinalTransform->SetParameters(finalParameters);

    // Print out results
    std::cout << "RESULTS:" << std::endl;
    std::cout << "   Iterations    = " << currentIterations << std::endl;
    std::cout << "   Metric value  = " << bestValue << std::endl;
    std::cout << "   Final transformation: " << std::endl;
    std::cout << m_FinalTransform << std::endl;

}

template <typename TransformationType, typename MetricType>
void RegisterImages<TransformationType, MetricType>
::WriteOutputImage(std::string outImageFile, ImageType::Pointer resampledImage)
{
  typedef itk::CastImageFilter<ImageType, ImageType> CastFilterType;
  typedef itk::ImageFileWriter<ImageType> WriterType;

  WriterType::Pointer writer = WriterType::New();
  CastFilterType::Pointer caster = CastFilterType::New();

  writer->SetFileName(outImageFile);
  caster->SetInput(resampledImage); //resampler->GetOutput()
  writer->SetInput(caster->GetOutput());
  writer->Update();

  std::cout << "\nOutput registered image file: " << writer->GetFileName() << "\n"
            << std::endl;
}

template <typename TransformationType, typename MetricType>
void RegisterImages<TransformationType, MetricType>
::WriteOutResults(std::string outImageFile)
{
    typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleFilterType;
    ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    resampler->SetTransform(m_FinalTransform);
    resampler->SetInput(m_MovingImage);
    resampler->SetSize(m_FixedImage->GetLargestPossibleRegion().GetSize());
    resampler->SetOutputOrigin(m_FixedImage->GetOrigin());
    resampler->SetOutputSpacing(m_FixedImage->GetSpacing());
    resampler->SetOutputDirection(m_FixedImage->GetDirection());
    resampler->SetDefaultPixelValue(m_BackgroundGrayLevel);

    itk::InterpolateImageFunction<ImageType, double>::Pointer interpolator;

    if (m_Interpolator == std::string("l"))
    {
        interpolator = itk::LinearInterpolateImageFunction<ImageType, double>::New();
        std::cout << "linear interpolation." << std::endl;
    }
    else if (m_Interpolator == std::string("n"))
    {
        interpolator = itk::NearestNeighborInterpolateImageFunction<ImageType, double>::New();
        std::cout << "nearest neighbor interpolation." << std::endl;
    }
    else
    {
        typedef itk::BSplineInterpolateImageFunction<ImageType, double> BSplineInterpolatorType;
        BSplineInterpolatorType::Pointer bsplineInterpolator = BSplineInterpolatorType::New();
        bsplineInterpolator->SetSplineOrder(4);

        std::cout << "B-Spline interpolation." << std::endl;
        std::cout << "The BSpline order: " << bsplineInterpolator->GetSplineOrder() << std::endl;

        interpolator = bsplineInterpolator;
    }

    resampler->SetInterpolator(interpolator);

    // I'm fixing the data type to double. Would probably
    // be smart to check the types and stick to the type
    // of the moving images.

   this->WriteOutputImage(outImageFile, resampler->GetOutput());


}
}

#endif
