;;; deq-mode-tests.el --- ERT tests for deq-mode -*- lexical-binding: t -*-

;; Run with:
;;   emacs -Q --batch -L . -l deq-mode-tests.el -f ert-run-tests-batch-and-exit

;;; Code:

(require 'ert)
(require 'imenu)
(require 'deq-mode)

;;; ── Helpers ───────────────────────────────────────────────────────────

(defmacro deq-test-with-buffer (text &rest body)
  "Insert TEXT into a temp buffer in `deq-mode', fontify, run BODY."
  (declare (indent 1) (debug t))
  `(with-temp-buffer
     (deq-mode)
     (insert ,text)
     (font-lock-ensure)
     (goto-char (point-min))
     ,@body))

(defun deq-test--face-at (regexp)
  "Search forward for REGEXP and return the face at its first character."
  (goto-char (point-min))
  (re-search-forward regexp)
  (get-text-property (match-beginning 0) 'face))

(defun deq-test--reindent (text)
  "Insert TEXT in a `deq-mode' buffer, strip indentation, then re-indent it.
Return the resulting buffer contents."
  (with-temp-buffer
    (deq-mode)
    (insert text)
    (goto-char (point-min))
    (while (re-search-forward "^[ \t]+" nil t) (replace-match ""))
    (indent-region (point-min) (point-max))
    (buffer-string)))

;;; ── Activation / derivation ───────────────────────────────────────────

(ert-deftest deq-mode/auto-mode-alist-triggers ()
  (with-temp-buffer
    (let ((buffer-file-name "/tmp/example.deq"))
      (set-auto-mode)
      (should (eq major-mode 'deq-mode)))))

(ert-deftest deq-mode/derives-from-prog-mode ()
  (with-temp-buffer
    (deq-mode)
    (should (derived-mode-p 'prog-mode))))

(ert-deftest deq-mode/sets-comment-syntax-vars ()
  (with-temp-buffer
    (deq-mode)
    (should (equal comment-start "# "))
    (should (string-match-p "#" comment-start-skip))))

;;; ── Syntax table ──────────────────────────────────────────────────────

(ert-deftest deq-mode/recognizes-line-comments ()
  (deq-test-with-buffer "GADGET Foo {\n  # a comment\n}\n"
    (re-search-forward "comment")
    (should (nth 4 (syntax-ppss)))))

(ert-deftest deq-mode/recognizes-strings ()
  (deq-test-with-buffer "IMPORT \"abc.deq\"\n"
    (re-search-forward "abc")
    (should (nth 3 (syntax-ppss)))))

(ert-deftest deq-mode/braces-form-sexps ()
  (deq-test-with-buffer "GADGET F { X 0 }\n"
    (re-search-forward "{")
    (backward-char)
    (let ((start (point)))
      (forward-sexp)
      (should (eq (char-before) ?\}))
      (should (> (point) start)))))

(ert-deftest deq-mode/brackets-form-sexps ()
  (deq-test-with-buffer "CODE Foo[[2,1,3]] {}\n"
    (re-search-forward "\\[\\[")
    (backward-char 2)
    (forward-sexp)
    (should (eq (char-before) ?\])))) ;; first ] of ]]

;;; ── Font-lock ─────────────────────────────────────────────────────────

(ert-deftest deq-mode/fontify-definition-keyword-and-name ()
  (deq-test-with-buffer "GADGET Foo {\n}\n"
    (should (eq (deq-test--face-at "GADGET") 'font-lock-keyword-face))
    (should (eq (deq-test--face-at "Foo")    'font-lock-function-name-face))))

(ert-deftest deq-mode/fontify-control-keyword-import ()
  (deq-test-with-buffer "IMPORT \"x.deq\"\n"
    (should (eq (deq-test--face-at "IMPORT") 'font-lock-keyword-face))))

(ert-deftest deq-mode/fontify-statement-keyword-readout ()
  (deq-test-with-buffer "GADGET F {\n  READOUT R0\n}\n"
    (should (eq (deq-test--face-at "READOUT") 'font-lock-builtin-face))))

(ert-deftest deq-mode/fontify-pauli-target ()
  (deq-test-with-buffer "GADGET F {\n  ERROR(0.1) X3 Y7 Z2\n}\n"
    (should (eq (deq-test--face-at "X3") 'deq-pauli-face))
    (should (eq (deq-test--face-at "Y7") 'deq-pauli-face))
    (should (eq (deq-test--face-at "Z2") 'deq-pauli-face))))

(ert-deftest deq-mode/fontify-check-target ()
  (deq-test-with-buffer "GADGET F {\n  CHECK C0 C12\n}\n"
    (should (eq (deq-test--face-at "C0")  'deq-check-face))
    (should (eq (deq-test--face-at "C12") 'deq-check-face))))

(ert-deftest deq-mode/fontify-readout-target ()
  (deq-test-with-buffer "GADGET F {\n  READOUT R5\n}\n"
    (should (eq (deq-test--face-at "R5") 'deq-readout-face))))

(ert-deftest deq-mode/fontify-logical-target ()
  (deq-test-with-buffer "GADGET F {\n  READOUT LZ3 LX1 LY2\n}\n"
    ;; Logical pattern must win over the bare-Pauli pattern.
    (should (eq (deq-test--face-at "LZ3") 'deq-logical-face))
    (should (eq (deq-test--face-at "LX1") 'deq-logical-face))
    (should (eq (deq-test--face-at "LY2") 'deq-logical-face))))

(ert-deftest deq-mode/fontify-ds-target ()
  (deq-test-with-buffer "GADGET F {\n  PROPAGATE LX0 FROM DS4\n}\n"
    (should (eq (deq-test--face-at "DS4") 'deq-readout-face))))

(ert-deftest deq-mode/fontify-record-target ()
  (deq-test-with-buffer "GADGET F {\n  CONDITIONAL rec[-1] LX0\n}\n"
    (should (eq (deq-test--face-at "rec\\[-1\\]") 'deq-record-face))))

(ert-deftest deq-mode/fontify-decorator ()
  (deq-test-with-buffer "@GTYPE(2)\nGADGET F {\n}\n"
    (should (eq (deq-test--face-at "@GTYPE") 'deq-decorator-face))))

(ert-deftest deq-mode/no-fontify-inside-comment ()
  (deq-test-with-buffer "GADGET F {\n  # X0 should not be a Pauli here\n}\n"
    (re-search-forward "X0")
    (let ((face (get-text-property (match-beginning 0) 'face)))
      ;; Inside a comment the face should be the comment face, not Pauli.
      (should (or (eq face 'font-lock-comment-face)
                  (and (listp face)
                       (memq 'font-lock-comment-face face))))
      (should-not (eq face 'deq-pauli-face)))))

;;; ── Indentation ───────────────────────────────────────────────────────

(ert-deftest deq-mode/indent-flat-gadget ()
  (let ((expected "GADGET Foo {\n    READOUT R0\n    CHECK C0\n}\n"))
    (should (equal (deq-test--reindent expected) expected))))

(ert-deftest deq-mode/indent-nested-repeat ()
  (let ((expected (concat
                   "GADGET Foo {\n"
                   "    REPEAT 3 {\n"
                   "        CX 0 1\n"
                   "    }\n"
                   "}\n")))
    (should (equal (deq-test--reindent expected) expected))))

(ert-deftest deq-mode/indent-respects-custom-offset ()
  (let ((deq-indent-offset 2)
        (expected (concat
                   "GADGET Foo {\n"
                   "  REPEAT 3 {\n"
                   "    CX 0 1\n"
                   "  }\n"
                   "}\n")))
    (should (equal (deq-test--reindent expected) expected))))

(ert-deftest deq-mode/indent-closing-brace-dedents ()
  ;; A line starting with `}' should sit one level out from its body.
  (with-temp-buffer
    (deq-mode)
    (insert "GADGET Foo {\n        READOUT R0\n        }\n")
    (goto-char (point-min))
    (forward-line 2)
    (deq-indent-line)
    (back-to-indentation)
    (should (= (current-column) 0))))

;;; ── Imenu ─────────────────────────────────────────────────────────────

(ert-deftest deq-mode/imenu-finds-all-definition-kinds ()
  (deq-test-with-buffer
      (concat
       "CODE C[[2,1]] {\n  LOGICAL X0 Z0\n}\n"
       "GADGET G {\n}\n"
       "COMPOSE M {\n}\n"
       "PROGRAM P {\n}\n")
    (let ((idx (imenu--make-index-alist)))
      (should (assoc "Codes"    idx))
      (should (assoc "Gadgets"  idx))
      (should (assoc "Compose"  idx))
      (should (assoc "Programs" idx))
      (should (assoc "C" (cdr (assoc "Codes"    idx))))
      (should (assoc "G" (cdr (assoc "Gadgets"  idx))))
      (should (assoc "M" (cdr (assoc "Compose"  idx))))
      (should (assoc "P" (cdr (assoc "Programs" idx)))))))

;;; ── Comment commands ─────────────────────────────────────────────────

(ert-deftest deq-mode/comment-region-uses-hash ()
  (with-temp-buffer
    (deq-mode)
    (insert "CX 0 1\n")
    (comment-region (point-min) (point-max))
    (goto-char (point-min))
    (should (looking-at "#"))))

(provide 'deq-mode-tests)

;;; deq-mode-tests.el ends here
