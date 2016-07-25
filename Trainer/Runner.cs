using ArtificialNeuralNetwork;
using ArtificialNeuralNetwork.ActivationFunctions;
using ArtificialNeuralNetwork.Factories;
using ArtificialNeuralNetwork.WeightInitializer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.GeneticAlgorithm;
using NeuralNetwork.GeneticAlgorithm.Evaluatable;
using NeuralNetwork.GeneticAlgorithm.Evolution;
using NeuralNetwork.GeneticAlgorithm.Utils;

namespace Trainer
{
    class Runner
    {
        static void Main(string[] args)
        {
            NeuralNetworkConfigurationSettings networkConfig = new NeuralNetworkConfigurationSettings
            {
                NumInputNeurons = 1,
                NumOutputNeurons = 1,
                NumHiddenLayers = 2,
                NumHiddenNeurons = 3,
                SummationFunction = new SimpleSummation(),
                ActivationFunction = new TanhActivationFunction()
            };
            GenerationConfigurationSettings generationSettings = new GenerationConfigurationSettings
            {
                UseMultithreading = true,
                GenerationPopulation = 1000
            };
            EvolutionConfigurationSettings evolutionSettings = new EvolutionConfigurationSettings
            {
                NormalMutationRate = 0.05,
                HighMutationRate = 0.5,
                GenerationsPerEpoch = 10,
                NumEpochs = 1000
            };
            MutationConfigurationSettings mutationSettings = new MutationConfigurationSettings
            {
                MutateAxonActivationFunction = true,
                MutateNumberOfHiddenLayers = true,
                MutateNumberOfHiddenNeuronsInLayer = true,
                MutateSomaBiasFunction = true,
                MutateSomaSummationFunction = true,
                MutateSynapseWeights = true
            };
            var random = new RandomWeightInitializer(new Random());
            INeuralNetworkFactory factory = NeuralNetworkFactory.GetInstance(SomaFactory.GetInstance(networkConfig.SummationFunction), AxonFactory.GetInstance(networkConfig.ActivationFunction), SynapseFactory.GetInstance(new RandomWeightInitializer(new Random()), AxonFactory.GetInstance(networkConfig.ActivationFunction)), SynapseFactory.GetInstance(new ConstantWeightInitializer(1.0), AxonFactory.GetInstance(new IdentityActivationFunction())), random);
            IBreeder breeder = BreederFactory.GetInstance(factory, random).Create();
            IMutator mutator = MutatorFactory.GetInstance(factory, random).Create(mutationSettings);
            IEvalWorkingSet history = EvalWorkingSetFactory.GetInstance().Create(50);
            IEvaluatableFactory evaluatableFactory = new GameEvaluationFactory();

            var GAFactory = GeneticAlgorithmFactory.GetInstance(evaluatableFactory);
            IGeneticAlgorithm evolver = GAFactory.Create(networkConfig, generationSettings, evolutionSettings, factory, breeder, mutator, history, evaluatableFactory);
            evolver.RunSimulation();
        }
    }
}
