using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkClasses;


namespace NeuralNetworkClasses.Tests
{
    [TestClass()]
    public class PictureConverterTests
    {
        [TestMethod()]
        public void ConvertTest()
        {
            //Arrange
            var converter = new PictureConverter();
            var inputs = converter.Convert(@"E:\Projects C#\Project1 - Lessons\FirstNeuralNetwork\NeuralNetworkClassesTests\Images\Parasitized.png");
            converter.Save("E:\\image.png", inputs);
            //Act

            //Assert
        }

    }
}