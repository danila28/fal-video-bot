-- The remix-from-reference flow was removed in favour of the AI Editor
-- (direct video editing instead of formula-based regeneration).
-- 007_remix_formulas.sql stays as an immutable historical record; this
-- migration retires the table it created.
DROP TABLE IF EXISTS remix_formulas;
