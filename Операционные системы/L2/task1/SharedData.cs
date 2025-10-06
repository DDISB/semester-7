using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace task1
{
    public static class SharedData
    {
        public static volatile bool[] flagDraw = new bool[2] { false, false };
        public static volatile int turnDraw = 0;

        public static volatile bool[] flagFile = new bool[2] { false, false };
        public static volatile int turnFile = 0;

        public static int scroll1 = 100;
        public static int scroll2 = 100;
    }
}
