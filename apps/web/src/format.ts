const diseaseLabels: Record<string, string> = {
  acute_diarrhoeal_disease: "Acute diarrhoeal disease",
  acute_gastroenteritis: "Acute gastroenteritis",
  aes_je: "AES / JE",
  chikungunya: "Chikungunya",
  cholera: "Cholera",
  dengue: "Dengue",
  malaria: "Malaria",
};

export function humanizeDisease(value: string | null | undefined): string | null {
  if (!value?.trim()) return null;
  const normalized = value.trim().toLowerCase().replaceAll(" ", "_");
  const known = diseaseLabels[normalized];
  if (known) return known;
  const readable = value.trim().replaceAll("_", " ");
  return readable.charAt(0).toUpperCase() + readable.slice(1);
}
