$("<div/>").addClass("footnotes").appendTo("body");

$("footnote").each(function(i) {
  var counter = i + 1;
  $("<sup>" + counter + "</sup>").attr("data-counter", counter).addClass('footnote-ref').insertBefore($(this));
  $(this).attr("data-counter", counter).detach().appendTo(".footnotes");
});

$(".footnote-ref")
  .click(function() {
    var counter = $(this).attr("data-counter");
    var y = $("footnote[data-counter=" + counter + "]").offset().top;
    $('body').animate({
        scrollTop: y
    }, 500);
  })
  .hover(function() {
    var counter = $(this).attr("data-counter");
    var html = $("footnote[data-counter=" + counter + "]").html();
    var element = $("<div/>").addClass("popup").html(html);

    element.appendTo($(this));
  }, function() {
    $(this).find(".popup").remove();
  });

$("cite")
  .click(function() {
    var cite = $(this);
    var id = cite.attr("href").substring(1);
    var y = $("#" + id).offset().top;
    $('body').animate({
        scrollTop: y
    }, 500);
  })
  .not(".table").not(".figure").hover(function() {
    var cite = $(this);
    var id = cite.attr("href").substring(1);
    var html = $("#" + id).html();
    var element = $("<div/>").addClass("popup").html(html);

    element.appendTo($(this));
  }, function() {
    $(this).find(".popup").remove();
  });

$("figure img, figure svg").click(function() {
  var html = $(this).get(0).outerHTML;
  var element = $("<div/>").addClass("lightbox").html(html);
  element.hide().appendTo("body").fadeIn();
});

$("body").on("click", ".lightbox", function() {
  $(this).fadeOut(300, function(){
    $(this).remove();
  });
});

