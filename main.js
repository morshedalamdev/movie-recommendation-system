// Input sample (the array I gave earlier)
const tasks = [
  [874, 4, 120],
  [129, 2, 45],
  [405, 9, 200],
  [33, 1, 30],
  [762, 7, 180],
  [561, 10, 60],
  [218, 3, 90],
  [997, 5, 150],
  [46, 2, 45],
  [684, 8, 210],
  [310, 6, 95],
  [571, 1, 100],
  [143, 4, 75],
  [809, 9, 240],
  [672, 7, 180],
  [27, 5, 20],
  [934, 10, 300],
  [400, 3, 85],
  [58, 2, 45],
  [286, 6, 110]
];

function sortedFunction(arr) {
  // work on a shallow copy so original array isn't mutated
  return arr.slice().sort((a, b) => {
    // a and b are [serial_num, priority, time]
    const [serialA, prioA, timeA] = a;
    const [serialB, prioB, timeB] = b;

    // 1) priority: higher priority first
    if (prioA !== prioB) return prioB - prioA;

    // 2) time: smaller time first
    if (timeA !== timeB) return timeA - timeB;

    // 3) serial_num: smaller serial first
    return serialA - serialB;
  });
}

// Example usage:
const sorted = sortedFunction(tasks);
console.log(JSON.stringify(sorted, null, 2));