;;; deq-mode.el --- Major mode for DEQ quantum error correction files -*- lexical-binding: t -*-

;; Copyright (C) 2026 Microsoft

;; Keywords: languages, quantum
;; Package-Requires: ((emacs "27.1"))
;; Version: 0.1.0

;; This file is not part of GNU Emacs.

;;; Commentary:

;; Major mode for editing `.deq' files — the source DSL for the DEQ
;; quantum error correction system.  Features:
;;
;;   * Syntax highlighting for all DEQ constructs (CODE, GADGET,
;;     COMPOSE, PROGRAM blocks; LOGICAL/STABILIZER declarations;
;;     INPUT/OUTPUT ports; READOUT/CHECK/ERROR/CONDITIONAL statements;
;;     embedded Stim instructions; decorators).
;;   * Distinct faces for the four target classes (Pauli, check,
;;     readout, logical Pauli shortcut) mirroring the VS Code
;;     extension's color scheme.
;;   * `{}'-aware indentation built on `syntax-ppss'.
;;   * Imenu navigation for CODE / GADGET / COMPOSE / PROGRAM names.
;;   * Auto-registers for files ending in `.deq'.
;;
;; Install:
;;
;;   (add-to-list 'load-path "/path/to/emacs-deq")
;;   (require 'deq-mode)

;;; Code:

(require 'prog-mode)

(defgroup deq nil
  "Major mode for DEQ quantum error correction files."
  :group 'languages
  :prefix "deq-")

(defcustom deq-indent-offset 4
  "Number of spaces per indentation level in `deq-mode'."
  :type 'integer
  :safe #'integerp
  :group 'deq)

;;; ── Faces ─────────────────────────────────────────────────────────────

(defface deq-pauli-face
  '((t :inherit font-lock-type-face))
  "Face for Pauli operators (e.g. `X0', `Y3', `Z7', `I2')."
  :group 'deq)

(defface deq-logical-face
  '((t :inherit font-lock-function-name-face))
  "Face for logical-Pauli shortcut targets (e.g. `LX0', `LY3', `LZ7')."
  :group 'deq)

(defface deq-check-face
  '((t :inherit font-lock-constant-face))
  "Face for check/detector targets (e.g. `C0', `C12')."
  :group 'deq)

(defface deq-readout-face
  '((t :inherit font-lock-variable-name-face))
  "Face for readout and detector-state targets (e.g. `R0', `DS3')."
  :group 'deq)

(defface deq-record-face
  '((t :inherit font-lock-variable-name-face))
  "Face for measurement-record and sweep-bit targets (e.g. `rec[-1]', `sweep[2]')."
  :group 'deq)

(defface deq-decorator-face
  '((t :inherit font-lock-preprocessor-face))
  "Face for decorators (e.g. `@GTYPE(2)')."
  :group 'deq)

;;; ── Keyword sets ──────────────────────────────────────────────────────

(defconst deq--definition-keywords
  '("CODE" "GADGET" "COMPOSE" "PROGRAM")
  "Top-level definition keywords that introduce a named block.")

(defconst deq--control-keywords
  '("IMPORT" "REPEAT")
  "Control-flow / file-level keywords.")

(defconst deq--statement-keywords
  '("INPUT" "OUTPUT"
    "LOGICAL" "STABILIZER"
    "READOUT" "OBSERVABLE_INCLUDE"
    "CHECK" "DETECTOR"
    "ERROR"
    "CONDITIONAL" "PRESELECT"
    "VIRTUAL" "PROPAGATE" "FROM" "FLIP"
    "ASSERT_EQ"
    "IN" "OUT")
  "Body-statement keywords used inside definitions.")

;;; ── Font-lock ─────────────────────────────────────────────────────────

(defconst deq-font-lock-keywords
  (let ((ctrl    (regexp-opt deq--control-keywords    'symbols))
        (stmts   (regexp-opt deq--statement-keywords  'symbols))
        (ident   "[A-Za-z][A-Za-z0-9_]*"))
    `(
      ;; Decorators: @Name (arguments highlighted normally).
      ("@[A-Za-z][A-Za-z0-9_]*" 0 'deq-decorator-face)

      ;; Definition introducers and their names:  CODE Foo / GADGET Foo / ...
      (,(concat "\\b\\(" (regexp-opt deq--definition-keywords) "\\)"
                "\\s-+\\(" ident "\\)")
       (1 font-lock-keyword-face)
       (2 font-lock-function-name-face))

      ;; Other control / statement keywords.
      (,ctrl  1 font-lock-keyword-face)
      (,stmts 1 font-lock-builtin-face)

      ;; Inverted-target prefix `!' (used before qubits or Paulis).
      ("!" 0 font-lock-negation-char-face)

      ;; QEC-specific targets — order matters: more specific first.
      ("\\<L[XYZ][0-9]+\\>"   0 'deq-logical-face)
      ("\\<DS[0-9]+\\>"       0 'deq-readout-face)
      ("\\<C[0-9]+\\>"        0 'deq-check-face)
      ("\\<R[0-9]+\\>"        0 'deq-readout-face)
      ("\\<[IXYZ][0-9]+\\>"   0 'deq-pauli-face)

      ;; Measurement record / sweep bit.
      ("rec\\[-[0-9]+\\]"     0 'deq-record-face)
      ("sweep\\[[0-9]+\\]"    0 'deq-record-face)

      ;; Numbers (integers and floats, with optional sign / exponent).
      ("\\_<-?\\(?:[0-9]+\\.?[0-9]*\\|\\.[0-9]+\\)\\(?:[eE][+-]?[0-9]+\\)?\\_>"
       0 font-lock-constant-face)))
  "Font-lock keywords for `deq-mode'.")

;;; ── Syntax table ──────────────────────────────────────────────────────

(defvar deq-mode-syntax-table
  (let ((st (make-syntax-table)))
    ;; `#' begins a line comment that ends at newline.
    (modify-syntax-entry ?#  "<"  st)
    (modify-syntax-entry ?\n ">"  st)
    ;; Identifiers may contain `_'.
    (modify-syntax-entry ?_  "w"  st)
    ;; Double-quoted strings.
    (modify-syntax-entry ?\" "\"" st)
    ;; Matched brackets — enables `forward-sexp', `show-paren-mode',
    ;; and `syntax-ppss' depth tracking used by the indenter.
    (modify-syntax-entry ?\( "()" st)
    (modify-syntax-entry ?\) ")(" st)
    (modify-syntax-entry ?\[ "(]" st)
    (modify-syntax-entry ?\] ")[" st)
    (modify-syntax-entry ?\{ "(}" st)
    (modify-syntax-entry ?\} "){" st)
    ;; `!' and `@' are punctuation, not word constituents — keeps
    ;; `forward-word' from absorbing them into adjacent identifiers.
    (modify-syntax-entry ?!  "."  st)
    (modify-syntax-entry ?@  "."  st)
    ;; `*' as punctuation (combiner in pauli products and stim targets).
    (modify-syntax-entry ?*  "."  st)
    st)
  "Syntax table for `deq-mode'.")

;;; ── Indentation ───────────────────────────────────────────────────────

(defun deq--paren-depth-at-bol ()
  "Return the unbalanced paren nesting depth at the start of the current line."
  (save-excursion
    (beginning-of-line)
    (car (syntax-ppss))))

(defun deq--line-starts-with-closer-p ()
  "Non-nil if the first non-whitespace char of the line is a closing bracket."
  (save-excursion
    (beginning-of-line)
    (skip-chars-forward " \t")
    (memq (char-after) '(?\} ?\) ?\]))))

(defun deq-indent-line ()
  "Indent the current line of DEQ source.

Uses `syntax-ppss' to compute the `{'/`('/`[' nesting depth at the
beginning of the line and indents to `depth * deq-indent-offset'.
Lines whose first non-whitespace character is a closer (`}', `)',
or `]') are dedented one level so they line up with their opener."
  (interactive)
  (let* ((depth   (deq--paren-depth-at-bol))
         (closer  (deq--line-starts-with-closer-p))
         (target  (* (max 0 (if closer (1- depth) depth))
                     deq-indent-offset))
         (at-or-before-text
          (<= (current-column) (current-indentation))))
    (if at-or-before-text
        (indent-line-to target)
      (save-excursion (indent-line-to target)))))

;;; ── Imenu ─────────────────────────────────────────────────────────────

(defconst deq-imenu-generic-expression
  (let ((ident "\\([A-Za-z][A-Za-z0-9_]*\\)"))
    `(("Codes"    ,(concat "^\\s-*CODE\\s-+"    ident) 1)
      ("Gadgets"  ,(concat "^\\s-*GADGET\\s-+"  ident) 1)
      ("Compose"  ,(concat "^\\s-*COMPOSE\\s-+" ident) 1)
      ("Programs" ,(concat "^\\s-*PROGRAM\\s-+" ident) 1)))
  "Imenu patterns for navigating DEQ definitions.")

;;; ── Mode definition ───────────────────────────────────────────────────

;;;###autoload
(define-derived-mode deq-mode prog-mode "DEQ"
  "Major mode for editing DEQ quantum error correction source files.

\\{deq-mode-map}"
  :syntax-table deq-mode-syntax-table
  :group 'deq
  (setq-local comment-start      "# ")
  (setq-local comment-start-skip "#+\\s-*")
  (setq-local comment-end        "")
  (setq-local comment-use-syntax t)
  (setq-local font-lock-defaults '(deq-font-lock-keywords))
  (setq-local indent-line-function #'deq-indent-line)
  (setq-local indent-tabs-mode nil)
  (setq-local imenu-generic-expression deq-imenu-generic-expression)
  (setq-local beginning-of-defun-function #'deq-beginning-of-defun)
  (setq-local end-of-defun-function       #'deq-end-of-defun))

(defconst deq--defun-start-regexp
  (concat "^\\(?:@[A-Za-z][A-Za-z0-9_]*[^\n]*\n\\s-*\\)*"
          "\\(?:CODE\\|GADGET\\|COMPOSE\\|PROGRAM\\)\\_>")
  "Regexp matching the start of a top-level DEQ definition (allowing decorators).")

(defun deq-beginning-of-defun (&optional arg)
  "Move backward to the beginning of the enclosing DEQ definition.
With ARG, repeat that many times (or forward if ARG is negative)."
  (let ((arg (or arg 1)))
    (if (> arg 0)
        (re-search-backward deq--defun-start-regexp nil 'move arg)
      (re-search-forward deq--defun-start-regexp nil 'move (- arg)))))

(defun deq-end-of-defun ()
  "Move to the end of the current DEQ definition.
Assumes point is on or before the opening `{' of the definition."
  (when (re-search-forward "{" nil 'move)
    (backward-char)
    (condition-case nil (forward-sexp) (scan-error nil))))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.deq\\'" . deq-mode))

(provide 'deq-mode)

;;; deq-mode.el ends here
