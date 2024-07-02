using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace ConsoleApp1
{
    
    enum CellAction {move, defend, attack, heal};

    public class cell
    {
        public Byte hp, full;
        public int x, y, step = 1;

        public int DoAction()
        {
            Random rnd = new Random();
            int action = rnd.Next(0, 5);
            return action;
        }

        public void Result(int newX, int newY, Byte newHp, Byte newFull)
        {
            this.x = newX;
            this.y = newY;
            this.hp = newHp;
        }

        public cell(int x, int y)
        {
            this.x = x;
            this.y = y;
            this.hp = 3;
            this.full = 3;
        }
    }

    public class CellFactory
    {
        private volatile static List<Tuple<int, int>> positions = new List<Tuple<int, int>>();
        
        public static List<cell> CreateCells(int cellsNum)
        {
            int cellsCount = cellsNum;
            List<cell> cells = new List<cell>();
            if (cellsCount > Program.mapW * Program.mapH)
            {
                cellsCount = Program.mapW * Program.mapH;
            }

            for (int i = 0; i < cellsCount; i++)
            {
                while(true)
                {
                    Random rnd = new Random();
                    int x = rnd.Next(0, Program.mapW);
                    int y = rnd.Next(0, Program.mapH);
                    Tuple<int, int> pos = new Tuple<int, int>(x, y);

                    if (!positions.Contains(pos))
                    {
                        positions.Add(pos);
                        cells.Add(new cell(x, y));
                        break;
                    }
                }
            }
            return cells; // Add this line to return the 'cells' list outside of the for loop.
        }
    }

    public class CellMap
    {
        public int[,] map;
        private int foods = 3; // 1/3 of the map has food

        public CellMap(int cmW, int cmH)
        {
            map = new int[cmW, cmH];
            for (int i = 0; i < cmW; i++)
            {
                for (int j = 0; j < cmH; j++)
                {
                    Random rnd = new Random();

                    // Only 1/2 of the map has food
                    int food = rnd.Next(0, 2) == 0 ? 2 : 0;
                    map[i, j] = food; 
                }
            }
        }
    }

    public class Game
    {
        internal CellMap cellMap;
        internal List<cell> cells;
        public volatile static int livingCellCount = 0;

        private Dictionary<Tuple<int, int>, List<cell>> positionToCells= new Dictionary<Tuple<int, int>, List<cell>>();

        public Game(CellMap cellMap, List<cell> cells)
        {
            this.cellMap = cellMap; // Ensure cellMap is assigned to handle non-nullable field warning
            this.cells = cells;
            livingCellCount = cells.Count;
        }

        // Process cell actions and update cellMap
        public void DoGameStep()
        {
            // Create a copy to store each cell's new position
            positionToCells.Clear();
            List<cell> nextCells = new List<cell>();

            foreach (cell cell in cells)
            {
                int action = cell.DoAction();
                if (cell.full > 0)
                    cell.full--;
                else
                    cell.hp--;
                switch (action)
                {
                    case 1: // y++
                        if (cell.y + 1 < cellMap.map.GetLength(1))
                        {
                            cell.y++;
                            if (positionToCells.ContainsKey(new Tuple<int, int>(cell.x, cell.y)))
                            {
                                positionToCells[new Tuple<int, int>(cell.x, cell.y)].Add(cell);
                            }
                            else
                            {
                                positionToCells.Add(new Tuple<int, int>(cell.x, cell.y), new List<cell> { cell });
                            }
                        }
                        break;
                    case 2: // y--
                        if (cell.y - 1 >= 0)
                        {
                            cell.y--;
                            if (positionToCells.ContainsKey(new Tuple<int, int>(cell.x, cell.y)))
                            {
                                positionToCells[new Tuple<int, int>(cell.x, cell.y)].Add(cell);
                            }
                            else
                            {
                                positionToCells.Add(new Tuple<int, int>(cell.x, cell.y), new List<cell> { cell });
                            }
                        }
                        break;
                    case 3: // x++
                        if (cell.x + 1 < cellMap.map.GetLength(0))
                        {
                            cell.x++;
                            if (positionToCells.ContainsKey(new Tuple<int, int>(cell.x, cell.y)))
                            {
                                positionToCells[new Tuple<int, int>(cell.x, cell.y)].Add(cell);
                            }
                            else
                            {
                                positionToCells.Add(new Tuple<int, int>(cell.x, cell.y), new List<cell> { cell });
                            }
                        }
                        break;
                    case 4: // x--
                        if (cell.x - 1 >= 0)
                        {
                            cell.x--;
                            if (positionToCells.ContainsKey(new Tuple<int, int>(cell.x, cell.y)))
                            {
                                positionToCells[new Tuple<int, int>(cell.x, cell.y)].Add(cell);
                            }
                            else
                            {
                                positionToCells.Add(new Tuple<int, int>(cell.x, cell.y), new List<cell> { cell });
                            }
                        }
                        break;
                    default:
                        if (cell.hp < 9)
                            cell.hp++;
                        if (positionToCells.ContainsKey(new Tuple<int, int>(cell.x, cell.y)))
                        {
                            positionToCells[new Tuple<int, int>(cell.x, cell.y)].Add(cell);
                        }
                        else
                        {
                            positionToCells.Add(new Tuple<int, int>(cell.x, cell.y), new List<cell> { cell });
                        }
                        break;
                }
            }

            foreach (List<cell> cells in positionToCells.Values)
            {
                cell livingCell = null;
                if (cells.Count > 1)
                {
                    livingCell = cells.Select(c => c).OrderByDescending(c => c.hp).First();
                    livingCell.hp = livingCell.hp = (byte)(cells.Sum(c => c.hp) / 2);
                    livingCell.full = livingCell.full = (byte)(cells.Sum(c => c.full) / 2);

                    if (livingCell.full > 9)
                    {
                        livingCell.full = 9;
                    }
                    if (livingCell.hp > 9)
                    {
                        livingCell.hp = 9;
                    }
                }
                else
                {
                    livingCell = cells.First();
                }

                nextCells.Add(livingCell);
            }

            foreach (cell cell in nextCells)
            {
                if (cellMap.map[cell.x, cell.y] > 0)
                {
                    cell.full += (byte)cellMap.map[cell.x, cell.y];
                    cell.hp += (byte)cellMap.map[cell.x, cell.y];
                    if (cell.full > 9)
                    {
                        cell.full = 9;
                    }
                    if (cell.hp > 9)
                    {
                        cell.hp = 9;
                    }
                    cellMap.map[cell.x, cell.y] = 0;
                }
            }
            nextCells.RemoveAll(c => c.hp <= 0);

            this.cells.Clear();
            this.cells = nextCells;
            livingCellCount = cells.Count;
        }
    }

    class Program
    {
        public static int mapW = 20;
        public static int mapH = 20;
        public static int cellCount = 20;

        static void Main(string[] args)
        {
            DateTime start = DateTime.Now;
            List<cell> cells = CellFactory.CreateCells(cellCount);
            CellMap cellMap = new CellMap(mapW, mapH);

            Game mv = new Game(cellMap, cells);

            int time = 0;
            int duration = 50;
            while (true)
            {
                mv.DoGameStep();
                GameUI.RefreshView(mv);
                Console.WriteLine(time);
                time++;
                if (Game.livingCellCount == 1)
                {
                    duration = 1000;
                }
                System.Threading.Thread.Sleep(duration);
                if (Game.livingCellCount == 0)
                {
                    break;
                }
            }
        }
    }
}
