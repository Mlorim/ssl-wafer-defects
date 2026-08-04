import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath =
  "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison.xlsx";
const previewPath =
  "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison_with_climex.png";

const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItemAt(0);
const usedBefore = sheet.getUsedRange();
const oldValues = JSON.stringify(usedBefore.values);
const oldFormulas = JSON.stringify(usedBefore.formulas);

if (!["A1:O8", "A1:O9"].includes(usedBefore.address)) {
  throw new Error(`Неожиданный диапазон таблицы: ${usedBefore.address}`);
}
const climexExists = usedBefore.values.some((row) => row?.[0] === "ClimEx (ResNet-18)");

if (!climexExists) {
  // Копируем только формат последней существующей строки в новую строку.
  sheet.getRange("A8:O8").copyTo(sheet.getRange("A9:O9"), "all");
  sheet.getRange("A9:O9").values = [[
  "ClimEx (ResNet-18)",
  "WM-811K; 10% labeled train (13 836) + 638 507 unlabeled; stratified 80/10/10; 96×96",
  null,
  null,
  null,
  0.842,
  0.9684880023128072,
  0.882781936102311,
  0.779646802436466,
  0.8195383779719039,
  null,
  null,
  null,
  null,
  "Основные причины: статья не задаёт параметры rotation, cutout и noise; неясно, считаются ли динамические пороги по всему unlabeled-набору или по окну итераций. У нас использовано окно 512 итераций, один seed вместо среднего по 10 запускам и постоянный LR.",
  ]];
  sheet.getRange("N9").formulas = [["=IF(OR(F9=\"\",J9=\"\"),\"\",J9-F9)"]];
}

// Форматируется исключительно новая строка, поскольку формат строк не
// расширяется автоматически за прежний used range при импорте XLSX.
sheet.getRange("A9:O9").format.rowHeight = 68;
sheet.getRange("A9:B9").format = { wrapText: true, verticalAlignment: "top" };
sheet.getRange("O9").format = { wrapText: true, verticalAlignment: "top" };
sheet.getRange("C9:N9").format = {
  numberFormat: "0.00%",
  horizontalAlignment: "right",
  verticalAlignment: "top",
};
sheet.getRange("N9").format.fill = "#FFE582";

// Существующие 8 строк должны остаться семантически неизменными. При
// пересчёте движок нормализует пустой результат формулы null -> "" и может
// раскрыть дополнительные двоичные знаки float; это не изменение ячейки.
const equivalent = (left, right) => {
  if ((left === null || left === "") && (right === null || right === "")) return true;
  if (typeof left === "number" && typeof right === "number") {
    return Math.abs(left - right) < 1e-12;
  }
  return left === right;
};
const beforeValues = JSON.parse(oldValues).slice(0, 8);
const afterValues = sheet.getRange("A1:O8").values;
for (let row = 0; row < beforeValues.length; row += 1) {
  for (let col = 0; col < beforeValues[row].length; col += 1) {
    if (!equivalent(beforeValues[row][col], afterValues[row][col])) {
      throw new Error(`Изменилась существующая ячейка row=${row + 1}, col=${col + 1}`);
    }
  }
}
const beforeFormulas = JSON.stringify(JSON.parse(oldFormulas).slice(0, 8));
if (JSON.stringify(sheet.getRange("A1:O8").formulas) !== beforeFormulas) {
  throw new Error("Изменились формулы существующих строк");
}

const check = await workbook.inspect({
  kind: "table",
  range: `${sheet.name}!A1:O9`,
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 15,
});
console.log(check.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: sheet.name,
  range: "A1:O9",
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(workbookPath);
console.log(workbookPath);
