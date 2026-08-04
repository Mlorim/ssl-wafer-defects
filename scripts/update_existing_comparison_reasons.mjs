import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath =
  "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison.xlsx";
const previewBeforePath =
  "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison_before.png";
const previewAfterPath =
  "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison_after.png";

const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItemAt(0);
const used = sheet.getUsedRange();

const before = await workbook.render({
  sheetName: sheet.name,
  range: used.address,
  scale: 1,
  format: "png",
});
await fs.writeFile(previewBeforePath, new Uint8Array(await before.arrayBuffer()));

const values = used.values;
console.log(JSON.stringify({ sheet: sheet.name, address: used.address, values }));
if (process.env.PREVIEW_ONLY === "1") {
  process.exit(0);
}

const reasonByModel = {
  "Baseline (ResNet-18)":
    "После устранения утечки и обучения 30 эпох итоговый macro-F1 = 78,79% против 78,87% в статье (−0,08 п.п.): результат практически воспроизведён. Большая разница в Accuracy связана с иным распределением test и сильным дисбалансом WM-811K.",
  "Mean Teacher":
    "Диагностический прогон остановлен после 26 эпох. Лучший validation macro-F1 = 70,22% на эпохе 7; позднее улучшения не было. Возможные дефекты: оценивается student вместо EMA-teacher, а teacher работает в режиме train, из-за чего нестабильны BatchNorm-статистики.",
};

const headerRow = values.findIndex(
  (row) => Array.isArray(row) && row.includes("Предполагаемые причины расхождения"),
);
if (headerRow < 0) {
  throw new Error("Не найден столбец причин расхождения");
}
const modelCol = values[headerRow].indexOf("Модель");
const reasonCol = values[headerRow].indexOf("Предполагаемые причины расхождения");
if (modelCol < 0 || reasonCol < 0) {
  throw new Error("Не найдены обязательные столбцы");
}

const updated = [];
for (let row = headerRow + 1; row < values.length; row += 1) {
  const model = values[row]?.[modelCol];
  if (!(model in reasonByModel)) continue;
  sheet.getCell(row, reasonCol).values = [[reasonByModel[model]]];
  updated.push(model);
}
console.log(JSON.stringify({ updated }));

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const after = await workbook.render({
  sheetName: sheet.name,
  range: used.address,
  scale: 1,
  format: "png",
});
await fs.writeFile(previewAfterPath, new Uint8Array(await after.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(workbookPath);
console.log(workbookPath);
