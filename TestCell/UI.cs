using ConsoleApp1;

using System.Drawing;

namespace ConsoleApp1
{
    public class GameUI
    {
        private volatile static string[,] view;

        private static ConsoleColor grassColor = ConsoleColor.Green;
        private static ConsoleColor cellColor = ConsoleColor.Red;

        public static string ConcatenateArray(string[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            string result = "";

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result += array[i, j];
                }
            }

            // 去掉最后一个多余的空格
            return result;
        }

        public static void RefreshView(Game game)
        {
            // Merge cellMap and cells into view
            view = new string[game.cellMap.map.GetLength(0) + 1, game.cellMap.map.GetLength(1) + 1];

            foreach (cell cell in game.cells)
            {
                view[cell.x, cell.y] = "C" + Convert.ToString(cell.hp); // Placeholder for cell
            }

            Console.Clear();

            for (int i = 0; i < game.cellMap.map.GetLength(0); i++)
            {
                for (int j = 0; j < game.cellMap.map.GetLength(1) + 1; j++)
                {
                    if (j == view.GetLength(1) - 1)
                    {
                        view[i, j] = "\r\n"; // Correctly place newline character at the end of each row
                        Console.ResetColor();
                        Console.Write(view[i, j]);
                    }
                    else if (string.IsNullOrWhiteSpace(view[i, j])) // When there is no cell at the current position
                    {
                        if ((game.cellMap.map[i, j] > 0))
                        {
                            view[i, j] = " " + Convert.ToString(game.cellMap.map[i, j]); // Placeholder for food
                            Console.ForegroundColor = grassColor;
                            Console.Write(view[i, j]);
                        }
                        else
                        {
                            view[i, j] = "  "; // Placeholder for empty cell
                            Console.ResetColor();
                            Console.Write(view[i, j]);
                        }
                    }
                    else // When there is a cell at the current position
                    {
                        Console.ForegroundColor = cellColor;
                        Console.Write(view[i, j]);
                    }
                }
            }

            string lastCellTrace = " ";
            if (game.cells.Count == 1)
                lastCellTrace = "HP : "  + game.cells.FirstOrDefault().hp + " FP : " + game.cells.FirstOrDefault().full;
            Console.WriteLine(lastCellTrace);
            //Console.WriteLine( ConcatenateArray(view) + lastCellTrace);
        }
    }
}
