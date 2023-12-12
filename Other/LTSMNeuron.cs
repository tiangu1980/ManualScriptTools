using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

class LSTMNeuron
{
    private int inputSize;
    private int hiddenSize;

    // LSTM参数
    private Matrix<double> weightInput;
    private Matrix<double> weightHidden;
    private Vector<double> biasInput;
    private Vector<double> biasHidden;

    // LSTM状态
    private Vector<double> cellState;
    private Vector<double> hiddenState;

    public LSTMNeuron(int inputSize, int hiddenSize)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        // 初始化权重和偏置
        weightInput = Matrix<double>.Build.Random(hiddenSize * 4, inputSize, new Normal());
        weightHidden = Matrix<double>.Build.Random(hiddenSize * 4, hiddenSize, new Normal());
        biasInput = Vector<double>.Build.Random(hiddenSize * 4, new Normal());
        biasHidden = Vector<double>.Build.Random(hiddenSize * 4, new Normal());

        // 初始化状态
        cellState = Vector<double>.Build.Dense(hiddenSize);
        hiddenState = Vector<double>.Build.Dense(hiddenSize);
    }

    // LSTM前向传播
    public Vector<double> Forward(Vector<double> input)
    {
        //var concatInput = input.ToColumnMatrix().Stack(cellState.ToColumnMatrix(), hiddenState.ToColumnMatrix());
        var concatInput = Matrix<double>.Build.DenseOfColumnVectors(new[] { input, cellState, hiddenState });

        // 计算LSTM的门控和新状态
        //var gates = (weightInput * concatInput + biasInput).PointwiseSigmoid();
        var biasInputMatrix = Matrix<double>.Build.Dense(biasInput.Count, concatInput.ColumnCount, (i, j) => biasInput[i]);
        //var gates = (weightInput * concatInput + biasInputMatrix).PointwiseSigmoid();
        var gates = (weightInput * concatInput + biasInputMatrix).Map(x => 1.0 / (1.0 + Math.Exp(-x)));

        //gates += (weightHidden * hiddenState.ToColumnMatrix() + biasHidden).PointwiseTanh();
        //gates = (weightHidden * hiddenState.ToColumnMatrix() + biasHidden).PointwiseTanh();
        var biasHiddenMatrix = Matrix<double>.Build.Dense(biasHidden.Count, hiddenState.Count, (i, j) => biasHidden[i]);
        gates = (weightHidden * hiddenState.ToColumnMatrix() + biasHiddenMatrix).PointwiseTanh();

        var inputGate = gates.SubMatrix(0, hiddenSize, 0, 1);
        var forgetGate = gates.SubMatrix(0, hiddenSize, 1, 1);
        var outputGate = gates.SubMatrix(0, hiddenSize, 2, 1);
        var cellGate = gates.SubMatrix(0, hiddenSize, 3, 1);

        //cellState = cellState.PointwiseMultiply(forgetGate) + inputGate.PointwiseMultiply(cellGate);
        cellState = cellState.PointwiseMultiply(cellGate.Column(0)) * inputGate.PointwiseMultiply(cellGate);

        //hiddenState = outputGate.PointwiseMultiply(cellState.PointwiseTanh());
        //hiddenState = outputGate.PointwiseMultiply(Matrix<double>.Build.DenseOfColumnVectors(cellState.PointwiseTanh()));
        //var stat = new[] { cellState.PointwiseTanh() };
        //hiddenState = outputGate.PointwiseMultiply(Matrix<double>.Build.DenseOfColumnVectors(stat));
        hiddenState = cellState.PointwiseTanh().PointwiseMultiply(outputGate.Column(0));



        return hiddenState;
    }

    // 简单的梯度下降训练
    public void Train(Vector<double> input, Vector<double> target, double learningRate)
    {
        // 前向传播
        var prediction = Forward(input);

        // 计算误差
        var error = target - prediction;

        // 反向传播
        // ... 这里需要根据LSTM的反向传播算法进行权重和偏置的更新
        // ... 这是一个简化版本，实际情况需要更复杂的实现

        // 示例：更新权重和偏置（实际中需要根据梯度下降等算法进行更新）
        weightInput += learningRate * error.OuterProduct(input);
        weightHidden += learningRate * error.OuterProduct(hiddenState);
        biasInput += learningRate * error;
        biasHidden += learningRate * error;
    }
}

class Program
{
    static void Main()
    {
        // 创建LSTM神经元，假设输入大小为5，隐藏状态大小为10
        LSTMNeuron lstmNeuron = new LSTMNeuron(5, 10);

        // 训练数据
        Matrix<double> trainingData = Matrix<double>.Build.DenseOfArray(new double[,]
        {
            {0.1, 0.2, 0.3, 0.4, 0.5},
            {0.2, 0.3, 0.4, 0.5, 0.6},
            {0.3, 0.4, 0.5, 0.6, 0.7},
            {0.4, 0.5, 0.6, 0.7, 0.8},
            // ... 这里放入20行5列的训练数据
        });

        // 训练LSTM神经元
        for (int epoch = 0; epoch < 1000; epoch++)
        {
            foreach (var row in trainingData.EnumerateRows())
            {
                Vector<double> input = row.SubVector(0, row.Count - 1);
                Vector<double> target = Vector<double>.Build.DenseOfArray(new[] { row[row.Count - 1] }); //row[row.Count - 1].ToColumnMatrix();

                lstmNeuron.Train(input, target, learningRate: 0.1);
            }
        }

        // 测试数据
        Matrix<double> testData = Matrix<double>.Build.DenseOfArray(new double[,]
        {
            // ... 这里放入5行5列的测试数据
        });

        // 测试LSTM神经元
        foreach (var row in testData.EnumerateRows())
        {
            Vector<double> input = row.SubVector(0, row.Count - 1);
            Vector<double> target = Vector<double>.Build.DenseOfArray(new[] { row[row.Count - 1] });//row[row.Count - 1].ToColumnMatrix();
            Vector<double> prediction = lstmNeuron.Forward(input);

            Console.WriteLine($"Target: {target}, Prediction: {prediction}");
        }
    }
}
