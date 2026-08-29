#ifndef GEMS_PYKVLCALCULATOR_H
#define GEMS_PYKVLCALCULATOR_H

#include "kvlAtlasMeshPositionCostAndGradientCalculator.h"
#include "kvlAtlasMeshToIntensityImageCostAndGradientCalculator.h"
#include "kvlAtlasMeshToIntensityImageLogDomainCostAndGradientCalculator.h"
#include "kvlConditionalGaussianEntropyCostAndGradientCalculator.h"
#include "kvlMutualInformationCostAndGradientCalculator.h"
#include "kvlAtlasMeshToPointSetCostAndGradientCalculator.h"
#include "kvlAverageAtlasMeshPositionCostAndGradientCalculator.h"
#include "kvlAtlasMeshToWishartGaussMixtureCostAndGradientCalculator.h"
#include "kvlAtlasMeshToFrobeniusGaussMixtureCostAndGradientCalculator.h"
#include "kvlAtlasMeshToDSWbetaGaussMixtureCostAndGradientCalculator.h"

#include "kvlAtlasMeshCollection.h"
#include "pyKvlImage.h"
#include "pyKvlNumpy.h"
#include "pybind11/pybind11.h"
#include "pyKvlMesh.h"

class KvlCostAndGradientCalculator {

public:
    kvl::AtlasMeshPositionCostAndGradientCalculator::Pointer calculator;

    KvlCostAndGradientCalculator(std::string typeName,
                                 std::vector<KvlImage> images,
                                 std::string boundaryCondition,
                                 KvlTransform transform=KvlTransform(nullptr),
                                 py::array_t<double> means=py::array_t<double>(),
                                 py::array_t<double> variances=py::array_t<double>(),
                                 py::array_t<float> mixtureWeights=py::array_t<float>(),
                                 py::array_t<int> numberOfGaussiansPerClass=py::array_t<int>(),
                                 py::array_t<double> targetPoints=py::array_t<double>()
    ){
        if (typeName == "AtlasMeshToIntensityImage" || typeName == "AtlasMeshToIntensityImageLogDomain" || \
            typeName == "DSWbeta" || typeName == "Frobenius" || typeName == "Wishart") {

            py::buffer_info means_info  = means.request();

            // Retrieve means if they are provided
            const int  numberOfGaussians = means_info.shape[0];
            const int  numberOfContrasts  = means_info.shape[1];

            std::vector< vnl_vector< double > >  means_converted;

            for ( int gaussianNumber = 0; gaussianNumber < numberOfGaussians; gaussianNumber++ ) {
                vnl_vector< double >  mean_converted( numberOfContrasts, 0.0f );

                for ( int contrastNumber = 0; contrastNumber < numberOfContrasts; contrastNumber++ ) {
                    mean_converted[ contrastNumber ] = means.at(gaussianNumber, contrastNumber);
                }
                means_converted.push_back( mean_converted );
            }
            // Retrieve variances if they are provided
            std::vector< vnl_matrix< double > >  variances_converted;
            for ( unsigned int gaussianNumber = 0; gaussianNumber < numberOfGaussians; gaussianNumber++ ) {
                vnl_matrix< double >  variance( numberOfContrasts, numberOfContrasts, 0.0f );
                for ( unsigned int row = 0; row < numberOfContrasts; row++ ) {
                  for ( unsigned int col = 0; col < numberOfContrasts; col++ ) {
                    variance[ row ][ col ] = variances.at(gaussianNumber, row, col);
                    }
                  }
                  variances_converted.push_back( variance );
            }
            // Retrieve mixtureWeights if they are provided
            std::vector< double >  mixtureWeights_converted  = std::vector< double >( numberOfGaussians, 0.0f );
            for ( int gaussianNumber = 0; gaussianNumber < numberOfGaussians; gaussianNumber++ ) {
                mixtureWeights_converted[gaussianNumber] = ( mixtureWeights.at(gaussianNumber));
            }
            // Retrieve numberOfGaussiansPerClass if they are provided
            const int  numberOfClasses = numberOfGaussiansPerClass.request().shape[0];
            std::vector< int >  numberOfGaussiansPerClass_converted = std::vector< int >( numberOfClasses, 0 );
            for ( int classNumber = 0; classNumber < numberOfClasses; classNumber++ ) {
                numberOfGaussiansPerClass_converted[ classNumber ] = numberOfGaussiansPerClass.at(classNumber);
            }

            //kvl::AtlasMeshToIntensityImageCostAndGradientCalculator::Pointer myCalculator;
            std::vector< ImageType::ConstPointer> images_converted;
            for(auto image: images){
                ImageType::ConstPointer constImage = static_cast< const ImageType* >( image.m_image.GetPointer() );
                images_converted.push_back( constImage );
            }
	    if (typeName == "AtlasMeshToIntensityImage")
            {
              kvl::AtlasMeshToIntensityImageCostAndGradientCalculator::Pointer myCalculator = kvl::AtlasMeshToIntensityImageCostAndGradientCalculator::New();
              myCalculator->SetImages( images_converted );
              myCalculator->SetParameters( means_converted, variances_converted, mixtureWeights_converted, numberOfGaussiansPerClass_converted );
              calculator = myCalculator;
            }
        
        else if (typeName == "DSWbeta")
            {
                
              kvl::AtlasMeshToDSWbetaGaussMixtureCostAndGradientCalculator::Pointer myCalculator = kvl::AtlasMeshToDSWbetaGaussMixtureCostAndGradientCalculator::New();
              myCalculator->SetImages( images_converted );
              myCalculator->SetParameters( means_converted, variances_converted, mixtureWeights_converted, numberOfGaussiansPerClass_converted );
              calculator = myCalculator;
            }
        else if (typeName == "Frobenius")
            {
              kvl::AtlasMeshToFrobeniusGaussMixtureCostAndGradientCalculator::Pointer myCalculator = kvl::AtlasMeshToFrobeniusGaussMixtureCostAndGradientCalculator::New();
              myCalculator->SetImages( images_converted );
              myCalculator->SetParameters( means_converted, variances_converted, mixtureWeights_converted, numberOfGaussiansPerClass_converted );
              calculator = myCalculator;
            }
        else if (typeName == "Wishart")
            {
              kvl::AtlasMeshToWishartGaussMixtureCostAndGradientCalculator::Pointer myCalculator = kvl::AtlasMeshToWishartGaussMixtureCostAndGradientCalculator::New();
              myCalculator->SetImages( images_converted );
              myCalculator->SetParameters( means_converted, variances_converted, mixtureWeights_converted, numberOfGaussiansPerClass_converted );
              calculator = myCalculator;
            }
            else
	        {
              kvl::AtlasMeshToIntensityImageLogDomainCostAndGradientCalculator::Pointer myCalculator = kvl::AtlasMeshToIntensityImageLogDomainCostAndGradientCalculator::New();
              myCalculator->SetImages( images_converted );
              myCalculator->SetParameters( means_converted, variances_converted, mixtureWeights_converted, numberOfGaussiansPerClass_converted );
              calculator = myCalculator;
            }
            /*
            std::vector< ImageType::ConstPointer> images_converted;
            for(auto image: images){
                ImageType::ConstPointer constImage = static_cast< const ImageType* >( image.m_image.GetPointer() );
                images_converted.push_back( constImage );
            }
            myCalculator->SetImages( images_converted );
            myCalculator->SetParameters( means_converted, variances_converted, mixtureWeights_converted, numberOfGaussiansPerClass_converted );
            calculator = myCalculator;
            */
        } else if (typeName == "MutualInformation") {

            kvl::MutualInformationCostAndGradientCalculator::Pointer myCalculator = kvl::MutualInformationCostAndGradientCalculator::New();
            myCalculator->SetImage( images[ 0 ].m_image );
            calculator = myCalculator;

        } else if (typeName == "PointSet") {

            kvl::AtlasMeshToPointSetCostAndGradientCalculator::Pointer myCalculator = kvl::AtlasMeshToPointSetCostAndGradientCalculator::New();
            kvl::AtlasMesh::PointsContainer::Pointer alphaTargetPoints = kvl::AtlasMesh::PointsContainer::New();
            CreatePointSetFromNumpy(alphaTargetPoints, targetPoints);
            myCalculator->SetTargetPoints( alphaTargetPoints );
            calculator = myCalculator;

        } else {
            throw std::invalid_argument("Calculator type not supported.");
        }

        // Specify the correct type of boundary condition
        switch( boundaryCondition[ 0 ] )
        {
            case 'S':
            {
                std::cout << "SLIDING" << std::endl;
                calculator->SetBoundaryCondition( kvl::AtlasMeshPositionCostAndGradientCalculator::SLIDING );

                // Retrieve transform if one is provided
                TransformType::ConstPointer  constTransform = nullptr;
                constTransform = static_cast< const TransformType* >( transform.m_transform.GetPointer() );

                if ( constTransform.GetPointer() )
                {
                    calculator->SetMeshToImageTransform( constTransform );
                }
                break;
            }
            case 'A':
            {
                std::cout << "AFFINE" << std::endl;
                calculator->SetBoundaryCondition( kvl::AtlasMeshPositionCostAndGradientCalculator::AFFINE );
                break;
            }
            case 'T':
            {
                std::cout << "TRANSLATION" << std::endl;
                calculator->SetBoundaryCondition( kvl::AtlasMeshPositionCostAndGradientCalculator::TRANSLATION );
                break;
            }
            case 'N':
            {
                std::cout << "NONE" << std::endl;
                calculator->SetBoundaryCondition( kvl::AtlasMeshPositionCostAndGradientCalculator::NONE );
                break;
            }
            default:
            {
                throw std::invalid_argument( "Boundary condition type not supported." );
            }
        }
    }

    KvlCostAndGradientCalculator(KvlMeshCollection meshCollection,
                                 double K0,
                                 double K1,
                                 KvlTransform transform
    ){
        kvl::AverageAtlasMeshPositionCostAndGradientCalculator::Pointer myCalculator = kvl::AverageAtlasMeshPositionCostAndGradientCalculator::New();
        myCalculator->SetBoundaryCondition(kvl::AtlasMeshPositionCostAndGradientCalculator::SLIDING);
        // Retrieve transform if provided
        TransformType::ConstPointer constTransform = static_cast<const TransformType*>(transform.m_transform.GetPointer());
        if (constTransform.GetPointer()) myCalculator->SetMeshToImageTransform( constTransform );
        // Apply positions and Ks
        kvl::AtlasMeshCollection::Pointer meshCollectionPtr = meshCollection.GetMeshCollection();
        std::vector<double> Ks = {K0};
        std::vector<kvl::AtlasMesh::PointsContainer::ConstPointer> positions = {meshCollectionPtr->GetReferencePosition()};
        for (int  meshNumber = 0; meshNumber < meshCollectionPtr->GetPositions().size(); meshNumber++) {
            positions.push_back(meshCollectionPtr->GetPositions()[meshNumber].GetPointer()); 
            Ks.push_back(K1);
        }
        myCalculator->SetPositionsAndKs(positions, Ks);
        calculator = myCalculator;
    }

    std::pair<double, py::array_t<double>> EvaluateMeshPosition(const KvlMesh &mesh) {
        calculator->Rasterize( mesh.mesh );
        const double cost = calculator->GetMinLogLikelihoodTimesPrior();
        kvl::AtlasPositionGradientContainerType::ConstPointer gradient = calculator->GetPositionGradient();

        const size_t numberOfNodes = gradient->Size();
        auto const data = new double[numberOfNodes*3];
        auto data_it = data;
        for ( kvl::AtlasPositionGradientContainerType::ConstIterator  it = gradient->Begin();
              it != gradient->End(); ++it )
        {
            *data_it++ = it.Value()[0];
            *data_it++ = it.Value()[1];
            *data_it++ = it.Value()[2];
        }
        py::array_t<double> gradient_np = createNumpyArrayCStyle({numberOfNodes, 3}, data);
        return {cost, gradient_np};
    };

    void SetDSWparams(int numberOfContrasts,
                 std::vector<KvlImage> DTIimages,
                 py::array_t< double >  DSWbetaMixtureWeights,
                 py::array_t< int >  numberOfDSWbetaePerClass,
                 double voxratio,
                 py::array_t< double > DSWbetaAlpha,
                 py::array_t< double > DSWbetaMeans,
                 py::array_t< double > DSWbetaBeta,
                 py::array_t< double > DSWbetaConcentration,
                 py::array_t< double > logKummerSamples,
                 double logKummerIncrement)
    {
        // convert DSWbetaMixWeights
        std::vector< double > DSWbetaMixtureWeights_converted;
        int DSWbetaWeightsToConvert = DSWbetaMixtureWeights.request().size;
        for (int i = 0; i <DSWbetaWeightsToConvert; i++){
            DSWbetaMixtureWeights_converted[ i ] = DSWbetaMixtureWeights.at(i);
        }

        // conver numberOfDSWbetaePerClass
        std::vector< int > numberOfDSWbetaePerClass_converted;
        int numberOfDSWbetaeToConvert = numberOfDSWbetaePerClass.request().size;
        for (int i = 0; i < numberOfDSWbetaeToConvert; i++){
            numberOfDSWbetaePerClass_converted[ i ] = numberOfDSWbetaePerClass.at(i);
        }

        //convet DSWbetaAlpha
        std::vector< double > DSWbetaAlpha_converted;
        int numDSWbetaAlphaToConvert = DSWbetaAlpha.request().size;
        for (int i = 0; i < numDSWbetaAlphaToConvert; i++){
            DSWbetaAlpha_converted[ i ] = DSWbetaAlpha.at(i);
        }   

        //convert DSWbetaMeans
        py::buffer_info DSWbetaMeans_info  = DSWbetaMeans.request();
        std::vector< vnl_vector< double >> DSWbetaMeans_converted;
        int numDSWMeansToConvert = DSWbetaMeans_info.shape[0];
        int meansToConvert = DSWbetaMeans_info.shape[1];
        for (int i = 0; i < numDSWMeansToConvert; i++){
            vnl_vector< double > betaMean_converted(meansToConvert, 0.0f);
            for (int j = 0; j < meansToConvert; j++){
                betaMean_converted[j] = DSWbetaMeans.at(i,j);
            }
            DSWbetaMeans_converted.push_back(betaMean_converted);
        }

        //convert DSWbetaBeta
        std::vector< double > DSWbetaBeta_converted;
        int DSWbetaBetaToConvert = DSWbetaBeta.request().size;
        for (int i = 0; i < DSWbetaBetaToConvert; i++){
            DSWbetaBeta_converted[ i ] = DSWbetaBeta.at(i);
        }

        // convert DSWbetaConcentration
        std::vector< double > DSWbetaConcentration_converted;
        int DSWbetaConcentrationToConvert = DSWbetaConcentration.request().size;
        for (int i = 0; i < DSWbetaConcentrationToConvert; i++){
            DSWbetaConcentration_converted[ i ] = DSWbetaConcentration.at(i);
        }

        //convert logKummerSamples
        std::vector< double > logKummerSamples_converted;
        int logKummerSamplesToConvert = logKummerSamples.request().size;
        for (int i = 0; i < logKummerSamplesToConvert; i++){
            logKummerSamples_converted[ i ] = logKummerSamples.at(i);
        }

        //convert DTIimages
        std::vector< ImageType::ConstPointer> DTIimages_converted;
        for(auto image: DTIimages){
            ImageType::ConstPointer constImage = static_cast< const ImageType* >( image.m_image.GetPointer() );
            DTIimages_converted.push_back( constImage );
        }

        calculator->SetDiffusionParameters(numberOfContrasts, DSWbetaMixtureWeights_converted, numberOfDSWbetaePerClass_converted, voxratio,\
                                           DSWbetaAlpha_converted, DSWbetaMeans_converted, DSWbetaBeta_converted, DSWbetaConcentration_converted,\
                                           logKummerSamples_converted, logKummerIncrement); 
        kvl::AtlasMeshToIntensityImageCostAndGradientCalculatorBase::Pointer  myCalculator= dynamic_cast< kvl::AtlasMeshToIntensityImageCostAndGradientCalculatorBase*>( calculator.GetPointer() );
        myCalculator->SetDiffusionImages(DTIimages_converted);
        
    }
    
    void SetFrobeniusParams(int numberOfContrasts,
                            std::vector<KvlImage> DTIimages,
                            py::array_t< double >  frobMixtureWeights,
                            py::array_t< int >  numberOfFrobeniusPerClass,
                            double voxratio,
                            py::array_t< double > frobVariance,
                            py::array_t< double > frobMeans)
    {
        // convert frobMixtureWeights
        std::vector< double > frobMixtureWeights_converted;
        int frobMixWeightsToConvert = frobMixtureWeights.request().size;
        for (int i = 0; i < frobMixWeightsToConvert; i++){
            frobMixtureWeights_converted[ i ] = frobMixtureWeights.at(i);
        }

        //convert numberOfFrobeniusPerClass
        std::vector< int > numberOfFrobeniusPerClass_converted;
        int numFrobPerClassToConvert = numberOfFrobeniusPerClass.request().size;
        for (int i = 0; i < numFrobPerClassToConvert; i++){
            numberOfFrobeniusPerClass_converted[ i ] = numberOfFrobeniusPerClass.at(i);
        }

        //convert frobVariance
        std::vector< double > frobVariance_converted;
        int frobVarToConvert = frobVariance.request().size;
        for (int i = 0; i < frobVarToConvert; i++){
            frobVariance_converted[ i ] = frobVariance.at(i);
        }

        //convert frobMeans
        std::vector< vnl_vector < double > > frobMeans_converted;
        int frobMeansToConvert = frobMeans.request().size;
        for (int i = 0; i < frobMeansToConvert; i++){
            vnl_vector< double > frobMean_converted(numFrobPerClassToConvert, 0.0f);
            //for (int j = 0; j < frobMeans.at(i).request().size; j++){
            for (int j = 0; j < frobMeansToConvert; j++){
                frobMean_converted[ j ] = frobMeans.at(i,j); 
            }
            frobMeans_converted.push_back(frobMean_converted);
        }

        //convert DTIimages
        std::vector< ImageType::ConstPointer> DTIimages_converted;
        for(auto image: DTIimages){
            ImageType::ConstPointer constImage = static_cast< const ImageType* >( image.m_image.GetPointer() );
            DTIimages_converted.push_back( constImage );
        }

        calculator->SetDiffusionParameters(numberOfContrasts, frobMixtureWeights_converted, numberOfFrobeniusPerClass_converted,\
                                           voxratio, frobVariance_converted, frobMeans_converted);
        kvl::AtlasMeshToIntensityImageCostAndGradientCalculatorBase::Pointer  myCalculator= dynamic_cast< kvl::AtlasMeshToIntensityImageCostAndGradientCalculatorBase*>( calculator.GetPointer() );
        myCalculator->SetDiffusionImages(DTIimages_converted);
    }
    
    void SetWishartParams(int numberOfContrasts,
                          std::vector<KvlImage> DTIimages,
                          py::array_t< double >  wmmMixtureWeights,
                          py::array_t< int >  numberOfWishartsPerClass,
                          double voxratio,
                          py::array_t< double > degreesOfFreedom,
                          py::array_t< double > scaleMatrices)
    {
        //convert wmmMixWeights
        std::vector< double > wmmMixtureWeights_converted;
        // should this be size_t?
        int numMixWeights = wmmMixtureWeights.request().size;
        for (int i = 0; i < numMixWeights; i++){
            wmmMixtureWeights_converted[ i ] = wmmMixtureWeights.at(i);
        }

        //convert numWisharts/class
        std::vector< int > numberOfWishartsPerClass_converted;
        int numWishartToConvert = numberOfWishartsPerClass.request().size;
        for (int i = 0; i < numWishartToConvert; i++){
            numberOfWishartsPerClass_converted[ i ] = numberOfWishartsPerClass.at(i);
        }

        //convert degFreedom
        std::vector< double > degreesOfFreedom_converted;
        int numToConvert = degreesOfFreedom.request().size;
        for (int i = 0; i < numToConvert; i++){
            degreesOfFreedom_converted[ i ] = degreesOfFreedom.at(i);
        }

        //convert scaleMatrices
        std::vector< vnl_matrix< double >> scaleMatrices_converted;
        int numMatToConvert = scaleMatrices.request().size;
        // loop through each element of scaleMatrices, each matrix should have dims of wmmMixWeights?
        for (int i = 0; i < numMatToConvert; i++){
            vnl_matrix< double > matrix_converted(numWishartToConvert, numWishartToConvert);
            for (int row = 0; row < numWishartToConvert; row++){
                for (int col = 0; col < numWishartToConvert; col++){
                    matrix_converted[ row ][ col ] = scaleMatrices.at(i, row, col);
                }
            }
        }

        //convert DTIimages
        std::vector< ImageType::ConstPointer> DTIimages_converted;
        for(auto image: DTIimages){
            ImageType::ConstPointer constImage = static_cast< const ImageType* >( image.m_image.GetPointer() );
            DTIimages_converted.push_back( constImage );
        }

        kvl::AtlasMeshToIntensityImageCostAndGradientCalculatorBase::Pointer  myCalculator= dynamic_cast< kvl::AtlasMeshToIntensityImageCostAndGradientCalculatorBase*>( calculator.GetPointer() );
        myCalculator->SetDiffusionImages(DTIimages_converted);
        calculator->SetDiffusionParameters(numberOfContrasts, wmmMixtureWeights_converted, numberOfWishartsPerClass_converted, voxratio, degreesOfFreedom_converted, scaleMatrices_converted);
    
    }
};

#endif //GEMS_PYKVLCALCULATOR_H
