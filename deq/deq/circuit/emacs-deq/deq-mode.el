;;; deq-mode.el --- Major mode for DEQ quantum error correction files -*- lexical-binding: t -*-

;; Copyright (C) 2026 Microsoft Corporation

;; Author: Juan M. Bello-Rivas <jbellorivas@microsoft.com>
;; URL: https://github.com/microsoft/qdk-ec
;; Keywords: languages
;; Package-Requires: ((emacs "30.1"))
;; Version: 0.2.0
;; SPDX-License-Identifier: MIT

;; This file is not part of GNU Emacs.

;;; Commentary:

;; Major mode for editing `.deq' files — the domain specific language for the
;; DEQ quantum error correction decoding system.  Features:
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

(defgroup deq nil
  "Major mode for DEQ quantum error correction files."
  :group 'languages
  :prefix "deq-")

(defcustom deq-indent-offset 4
  "Number of spaces per indentation level in `deq-mode'."
  :type 'integer
  :safe #'integerp)

;;; ── Faces ─────────────────────────────────────────────────────────────

(defface deq-pauli-face
  '((t :inherit font-lock-type-face))
  "Face for Pauli operators (e.g. `X0', `Y3', `Z7', `I2').")

(defface deq-logical-face
  '((t :inherit font-lock-function-name-face))
  "Face for logical-Pauli shortcut targets (e.g. `LX0', `LY3', `LZ7').")

(defface deq-check-face
  '((t :inherit font-lock-constant-face))
  "Face for check/detector targets (e.g. `C0', `C12').")

(defface deq-readout-face
  '((t :inherit font-lock-variable-name-face))
  "Face for readout and detector-state targets (e.g. `R0', `DS3').")

(defface deq-record-face
  '((t :inherit font-lock-variable-name-face))
  "Face for measurement-record and sweep-bit targets (e.g. `rec[-1]', `sweep[2]').")

(defface deq-decorator-face
  '((t :inherit font-lock-preprocessor-face))
  "Face for decorators (e.g. `@GTYPE(2)').")

;;; ── Regexp building blocks ────────────────────────────────────────────

(rx-define deq--identifier
  (seq (in "A-Za-z") (* (in "A-Za-z0-9_"))))

(rx-define deq--number
  (seq (? "-")
       (or (seq (+ digit) (? "." (* digit))) (seq "." (+ digit)))
       (? (in "eE") (? (in "+-")) (+ digit))))

(rx-define deq--definition-keyword
  (or "CODE" "GADGET" "COMPOSE" "PROGRAM"))

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

;;; ── Syntax ────────────────────────────────────────────────────────────

(defvar deq-mode-syntax-table
  (let ((table (make-syntax-table)))
    (modify-syntax-entry ?#  "<"  table)
    (modify-syntax-entry ?\n ">"  table)
    (modify-syntax-entry ?_  "w"  table)
    (modify-syntax-entry ?\" "\"" table)
    (modify-syntax-entry ?\( "()" table)
    (modify-syntax-entry ?\) ")(" table)
    (modify-syntax-entry ?\[ "(]" table)
    (modify-syntax-entry ?\] ")[" table)
    (modify-syntax-entry ?\{ "(}" table)
    (modify-syntax-entry ?\} "){" table)
    (modify-syntax-entry ?!  "."  table)
    (modify-syntax-entry ?@  "."  table)
    (modify-syntax-entry ?*  "."  table)
    table)
  "Syntax table for `deq-mode'.")

;;; ── Font-lock ─────────────────────────────────────────────────────────

(defconst deq-font-lock-keywords
  `((,(rx "@" deq--identifier) 0 'deq-decorator-face)

    (,(rx word-boundary (group deq--definition-keyword) (+ space) (group deq--identifier))
     (1 'font-lock-keyword-face)
     (2 'font-lock-function-name-face))
    (,(regexp-opt deq--control-keywords 'symbols) 0 'font-lock-keyword-face)
    (,(regexp-opt deq--statement-keywords 'symbols) 0 'font-lock-builtin-face)

    ("!" 0 'font-lock-negation-char-face)

    ;; More specific targets first: the first match wins.
    (,(rx bow "L" (in "XYZ") (+ digit) eow) 0 'deq-logical-face)
    (,(rx bow "DS" (+ digit) eow) 0 'deq-readout-face)
    (,(rx bow "C" (+ digit) eow) 0 'deq-check-face)
    (,(rx bow "R" (+ digit) eow) 0 'deq-readout-face)
    (,(rx bow (in "IXYZ") (+ digit) eow) 0 'deq-pauli-face)
    (,(rx (or (seq "rec[-" (+ digit) "]") (seq "sweep[" (+ digit) "]")))
     0 'deq-record-face)
    (,(rx symbol-start deq--number symbol-end) 0 'font-lock-constant-face))
  "Font-lock keywords for `deq-mode'.")

;;; ── Indentation ───────────────────────────────────────────────────────

(defun deq-indent-line ()
  "Indent the current line of DEQ source.

Indent to the `{'/`('/`[' nesting depth times `deq-indent-offset',
dedenting lines that start with a closer by one level."
  (interactive)
  (let* ((depth (car (syntax-ppss (pos-bol))))
         (closer (save-excursion
                   (back-to-indentation)
                   (memq (char-after) '(?\} ?\) ?\]))))
         (target (* (max 0 (if closer (1- depth) depth)) deq-indent-offset)))
    (if (<= (current-column) (current-indentation))
        (indent-line-to target)
      (save-excursion (indent-line-to target)))))

;;; ── Navigation ────────────────────────────────────────────────────────

(defconst deq-imenu-generic-expression
  `(("Codes"    ,(rx bol (* blank) "CODE"    (+ blank) (group deq--identifier)) 1)
    ("Gadgets"  ,(rx bol (* blank) "GADGET"  (+ blank) (group deq--identifier)) 1)
    ("Compose"  ,(rx bol (* blank) "COMPOSE" (+ blank) (group deq--identifier)) 1)
    ("Programs" ,(rx bol (* blank) "PROGRAM" (+ blank) (group deq--identifier)) 1))
  "Imenu patterns for navigating DEQ definitions.")

(defun deq--search-definition (backward)
  "Move to the previous (if BACKWARD) or next top-level definition.
Point ends at its first decorator, if any.  Return non-nil if found."
  (let ((keyword-regexp (rx bol (* blank) deq--definition-keyword symbol-end))
        (found nil))
    (unless backward
      (while (looking-at-p (rx (* blank) "@"))
        (forward-line 1))
      (end-of-line))
    (while (and (not found)
                (if backward
                    (re-search-backward keyword-regexp nil 'move)
                  (re-search-forward keyword-regexp nil 'move)))
      (let ((state (save-excursion (syntax-ppss (match-beginning 0)))))
        (setq found (and (zerop (car state)) (not (nth 8 state))))))
    (when found
      (goto-char (match-beginning 0))
      (while (and (not (bobp))
                  (save-excursion
                    (forward-line -1)
                    (looking-at-p (rx (* blank) "@"))))
        (forward-line -1)))
    found))

(defun deq-beginning-of-defun (&optional arg)
  "Move backward to the beginning of the enclosing DEQ definition.
With ARG, repeat that many times (or forward if ARG is negative)."
  (let ((arg (or arg 1))
        (found t))
    (dotimes (_ (abs arg))
      (setq found (deq--search-definition (> arg 0))))
    found))

(defun deq-end-of-defun ()
  "Move to the end of the current DEQ definition.
Assumes point is on or before the opening `{' of the definition."
  (when (re-search-forward "{" nil 'move)
    (backward-char)
    (ignore-error scan-error (forward-sexp))))

;;; ── Mode definition ───────────────────────────────────────────────────

;;;###autoload
(define-derived-mode deq-mode prog-mode "DEQ"
  "Major mode for editing DEQ quantum error correction source files.

\\{deq-mode-map}"
  (setq-local comment-start "# ")
  (setq-local comment-start-skip "#+\\s-*")
  (setq-local comment-end "")
  (setq-local comment-use-syntax t)
  (setq-local font-lock-defaults '(deq-font-lock-keywords))
  (setq-local indent-line-function #'deq-indent-line)
  (setq-local indent-tabs-mode nil)
  (setq-local imenu-generic-expression deq-imenu-generic-expression)
  (setq-local beginning-of-defun-function #'deq-beginning-of-defun)
  (setq-local end-of-defun-function #'deq-end-of-defun))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.deq\\'" . deq-mode))

(provide 'deq-mode)

;;; deq-mode.el ends here
